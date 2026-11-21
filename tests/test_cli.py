from agentdataset.cli import main


def test_cli_help():
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code in (0, None)


def test_cli_subcommand_help():
    for cmd in ("discover", "run", "review"):
        try:
            main([cmd, "--help"])
        except SystemExit as e:
            assert e.code in (0, None)