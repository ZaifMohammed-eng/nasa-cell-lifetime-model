def test_pipeline_compiles(tmp_path):
    from kfp import compiler
    from pipelines.kfp_pipeline import training_pipeline
    out = tmp_path / "p.yaml"
    compiler.Compiler().compile(training_pipeline, str(out))
    text = out.read_text()
    for step in ("data-check", "tune-evaluate-register", "shift-check", "release-note"):
        assert step in text
    assert "install_kfp" not in text and "pip install" not in text  # image already has kfp
