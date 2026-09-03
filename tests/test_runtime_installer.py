import pytest

from app.core.errors import ValidationError
from app.hardware.runtime_detector import InferenceRuntime
from app.hardware.runtime_installer import RuntimeInstaller


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _installer(system="Darwin", binaries=("brew",), runner=None) -> RuntimeInstaller:
    return RuntimeInstaller(
        run_command=runner or (lambda args: (0, "ok")),
        system=system,
        has_binary=lambda name: name in binaries,
    )


def test_macos_offer_uses_homebrew_when_available():
    offer = _installer().offer(InferenceRuntime.OLLAMA)

    assert offer.supported is True
    assert offer.command == ["brew", "install", "ollama"]
    assert "ollama.com" in offer.documentation_url


def test_macos_without_homebrew_points_to_the_website():
    offer = _installer(binaries=()).offer(InferenceRuntime.OLLAMA)

    assert offer.supported is False
    assert offer.command is None
    assert "Homebrew" in offer.note


def test_linux_offer_uses_the_official_install_script():
    offer = _installer(system="Linux", binaries=("sh", "curl")).offer(
        InferenceRuntime.OLLAMA
    )

    assert offer.supported is True
    assert "install.sh" in offer.manual_command


def test_unsupported_system_offers_no_automatic_installation():
    offer = _installer(system="Windows", binaries=()).offer(InferenceRuntime.OLLAMA)

    assert offer.supported is False
    assert offer.automatic_install_enabled is False


def test_vllm_is_never_installed_automatically():
    offer = _installer().offer(InferenceRuntime.VLLM)

    assert offer.supported is False
    assert offer.automatic_install_enabled is False
    assert "CUDA" in offer.note


def test_installation_without_confirmation_is_rejected():
    with pytest.raises(ValidationError, match="подтверждения"):
        _installer().install(InferenceRuntime.OLLAMA, confirmed=False)


def test_installation_is_refused_while_disabled_by_configuration(monkeypatch):
    monkeypatch.setenv("RUNTIME_INSTALL_ENABLED", "false")

    with pytest.raises(ValidationError, match="недоступна"):
        _installer().install(InferenceRuntime.OLLAMA, confirmed=True)


def test_confirmed_installation_runs_the_command(monkeypatch):
    monkeypatch.setenv("RUNTIME_INSTALL_ENABLED", "true")
    calls: list[list[str]] = []

    def runner(args):
        calls.append(args)
        return 0, "installed"

    result = _installer(runner=runner).install(InferenceRuntime.OLLAMA, confirmed=True)

    assert calls == [["brew", "install", "ollama"]]
    assert result.succeeded is True
    assert result.output == "installed"


def test_failed_installation_is_reported_without_raising(monkeypatch):
    monkeypatch.setenv("RUNTIME_INSTALL_ENABLED", "true")

    result = _installer(runner=lambda args: (1, "brew: command failed")).install(
        InferenceRuntime.OLLAMA, confirmed=True
    )

    assert result.succeeded is False
    assert "failed" in result.output


def test_nothing_is_executed_while_only_building_an_offer():
    def fail(args):  # pragma: no cover - не должен вызываться
        raise AssertionError("offer не имеет права ничего выполнять")

    _installer(runner=fail).offer(InferenceRuntime.OLLAMA)
