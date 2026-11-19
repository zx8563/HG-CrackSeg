from pathlib import Path
import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from core.checkpoint import load_checkpoint, normalize_state_dict
from core.losses import get_loss_function
from core.metrics import binary_image_metrics
from core.prediction_artifacts import save_prediction_artifacts
from data.evaluation_dataset import dataset_specs, EvaluationDataset, index_pairs
from models import create_model
from models.hg_crackseg import LargeKernelResidualBlock

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def configuration():
    return yaml.safe_load((ROOT / "configs/hg_crackseg.yaml").read_text())


def make_pairs(root, count=2, size=64):
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(count):
        image = np.random.default_rng(i).integers(0, 256, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), np.uint8)
        mask[10:50, 20 + i : 23 + i] = 255
        Image.fromarray(image).save(root / "images" / f"{i}.png")
        Image.fromarray(mask).save(root / "masks" / f"{i}.png")


def test_checkpoint_conversion_and_rejection(tmp_path):
    config = configuration()
    config["model"]["base_c"] = 4
    model = create_model(config["model"])
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    legacy = {
        "module."
        + k.replace("down3_large_kernel.", "down3_mamba.").replace(
            "down4_large_kernel.", "down4_mamba."
        ): v
        for k, v in expected.items()
    }
    path = tmp_path / "weights.pth"
    torch.save({"state_dict": legacy}, path)
    result = load_checkpoint(model, path)
    assert result["strict"] and result["renamed_keys"]
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    duplicate = {
        "down3_mamba.dwconv.weight": torch.zeros(1),
        "down3_large_kernel.dwconv.weight": torch.zeros(1),
    }
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_state_dict(duplicate)
    broken = dict(legacy)
    broken.pop(next(iter(broken)))
    torch.save(broken, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(model, path)


def test_evaluation_rejects_saved_configuration_drift(tmp_path):
    from evaluate_robustness import validate_preprocessing

    original = configuration()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(original))
    entry = dict(
        id="hg_crackseg",
        checkpoint=str(tmp_path / "weights.pth"),
        config=str(tmp_path / "config.yaml"),
        normalization="imagenet",
    )
    assert validate_preprocessing({}, entry, original) == "imagenet"
    changed = copy.deepcopy(original)
    changed["model"]["hessian_residual_scale"] = 0
    with pytest.raises(ValueError, match="hessian_residual_scale"):
        validate_preprocessing({}, entry, changed)


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs/ablations").glob("*.yaml")), ids=lambda p: p.stem
)
def test_ablation_forward_backward(path):
    config = yaml.safe_load(path.read_text())
    model = create_model(config["model"])
    criterion = get_loss_function(config["loss"]["name"], **config["loss"]["kwargs"])
    logits = model(torch.randn(2, 3, 32, 32))
    loss, _ = criterion(logits, torch.randint(0, 2, (2, 1, 32, 32)))
    assert logits["body_logits"].shape == (2, 2, 32, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_model_contract_and_protocol():
    config = configuration()
    model = create_model(config["model"])
    assert sum(p.numel() for p in model.parameters()) == 4874529
    assert isinstance(model.down3_large_kernel, LargeKernelResidualBlock)
    assert isinstance(model.down4_large_kernel, LargeKernelResidualBlock)
    specs = dataset_specs(ROOT, {})
    assert len(specs) == 19 and specs[0]["dataset_id"] == "clean"
    assert len({s["dataset_id"] for s in specs}) == 19
    assert all(s["dataset_kind"] != "external" for s in specs)
    assert len(dataset_specs(ROOT, {"include_corruptions": False})) == 1
    with pytest.raises(ValueError):
        create_model({"architecture": "hg_crackseg", "pretrained": True})


def test_metrics_and_export_use_same_binary_mask(tmp_path):
    target = np.zeros((64, 64), bool)
    target[10:50, 20:23] = True
    prediction = target.copy()
    prediction[30:50] = False
    metrics = binary_image_metrics(prediction, target)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(1)
    assert metrics["recall"] == pytest.approx(0.5)
    image = tmp_path / "source.jpg"
    Image.new("RGB", (64, 64)).save(image)
    artifact = save_prediction_artifacts(tmp_path / "export", image, prediction, target)
    with Image.open(tmp_path / "export" / artifact["predicted_mask"]) as opened:
        assert np.array_equal(np.asarray(opened) > 127, prediction)
    with Image.open(tmp_path / "export" / artifact["comparison"]) as opened:
        assert opened.size == (192, 88)


def test_generator_resume_preserves_pixels(tmp_path):
    from generate_corruptions import generate, parse_args

    root, output = tmp_path / "source", tmp_path / "corruptions"
    make_pairs(root / "Test")
    args = parse_args(["--root", str(root), "--out", str(output)])
    generate(args)
    manifest = json.loads((output / "corruption_manifest.json").read_text())
    assert manifest["completed"] and len(manifest["outputs"]) == 36
    missing = output / manifest["outputs"][1]["image"]
    missing.unlink()
    missing.with_suffix(".png.tmp").write_bytes(b"interrupted write")
    generate(args)
    repeated = json.loads((output / "corruption_manifest.json").read_text())
    assert repeated == manifest
    for row in repeated["outputs"]:
        with Image.open(output / row["mask"]) as actual, Image.open(
            root / "Test/masks" / row["source"]
        ) as expected:
            assert np.array_equal(np.asarray(actual), np.asarray(expected))
    args.seed += 1
    with pytest.raises(ValueError, match="protocol"):
        generate(args)


def test_epoch_resume_matches_uninterrupted_training(tmp_path):
    import train

    data = tmp_path / "data"
    for split in ("Train", "Validation"):
        make_pairs(data / split)
    config = configuration()
    config["model"]["base_c"] = 4
    config["data"].update(root_dir=str(data), batch_size=2, num_workers=0)
    config["training"].update(epochs=2, device="cpu", operator_freeze_epochs=1, save_interval=0)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    assert train.main(["--config", str(path), "--output-dir", str(continuous)]) == 0
    assert train.main(["--config", str(path), "--output-dir", str(resumed), "--epochs", "1"]) == 0
    checkpoint = resumed / "last_checkpoint.pth"
    assert (
        train.main(
            ["--config", str(path), "--output-dir", str(resumed), "--resume", str(checkpoint)]
        )
        == 0
    )
    a = torch.load(continuous / "last_checkpoint.pth", weights_only=True)
    b = torch.load(checkpoint, weights_only=True)
    assert a["epoch"] == b["epoch"] == 2
    assert a["best_dice"] == b["best_dice"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["scheduler"] == b["scheduler"]
    config["training"]["learning_rate"] *= 2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="recipe"):
        train.main(
            ["--config", str(path), "--output-dir", str(resumed), "--resume", str(checkpoint)]
        )
