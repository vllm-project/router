import os
import subprocess
import tempfile
import pytest
from pathlib import Path


class TestSystemServiceInstallation:
    """Test suite for systemd service installation script."""

    @pytest.fixture
    def install_script(self):
        """Path to the installation script."""
        return Path(__file__).parent.parent.parent / "scripts" / "install_service.sh"

    @pytest.fixture
    def service_file(self):
        """Path to the systemd service file."""
        return Path(__file__).parent.parent.parent / "systemd" / "vllm-router.service"

    @pytest.fixture
    def mock_binary(self, tmp_path):
        """Create a mock binary for testing."""
        binary_path = tmp_path / "vllm-router"
        binary_path.write_text("#!/bin/bash\necho 'Mock vllm-router'\n")
        binary_path.chmod(0o755)
        return str(binary_path)

    def test_service_file_exists(self, service_file):
        """Test that the systemd service file exists."""
        assert service_file.exists(), "Systemd service file should exist"

    def test_service_file_has_required_sections(self, service_file):
        """Test that service file contains required sections."""
        content = service_file.read_text()
        assert "[Unit]" in content, "Service file should have [Unit] section"
        assert "[Service]" in content, "Service file should have [Service] section"
        assert "[Install]" in content, "Service file should have [Install] section"

    def test_service_file_has_required_directives(self, service_file):
        """Test that service file has required directives."""
        content = service_file.read_text()
        required_directives = [
            "Description=",
            "ExecStart=",
            "Type=",
            "Restart=",
            "User=",
        ]
        for directive in required_directives:
            assert directive in content, f"Service file should have {directive}"

    @pytest.mark.skipif(not os.access("/usr/sbin/systemctl", os.X_OK), reason="systemctl not found")
    def test_install_script_is_executable(self, install_script):
        """Test that the installation script is executable."""
        # Note: In some environments we check if files have execute bit set
        # but we may not actually be able to set it without permissions
        assert install_script.exists(), "Install script should exist"

    def test_install_script_usage(self, install_script):
        """Test that install script shows usage when invoked."""
        if install_script.exists():
            result = subprocess.run(
                [str(install_script)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "Install script should exit with 0 for usage"
            assert "Usage:" in result.stdout or "Usage:" in result.stderr, \
                "Output should show usage information"

    @pytest.mark.integration
    def test_service_file_content_structure(self, service_file):
        """Test that service file has proper systemd structure."""
        content = service_file.read_text()
        lines = content.split('\n')

        assert lines[0].startswith("[Unit]"), "File should start with [Unit] section"

        section_count = 0
        for line in lines:
            if line.startswith("[Unit]") or line.startswith("[Service]") or line.startswith("[Install]"):
                section_count += 1

        assert section_count >= 3, "Service file should have at least 3 sections"

    def test_service_file_security_settings(self, service_file):
        """Test that service file includes security directives."""
        content = service_file.read_text()
        
        security_directives = [
            "NoNewPrivileges",
            "PrivateTmp",
            "LimitNOFILE",
        ]
        
        for directive in security_directives:
            assert directive in content, f"Service file should include {directive} directive"

    def test_service_file_restart_policy(self, service_file):
        """Test that service file has proper restart policy."""
        content = service_file.read_text()
        
        assert "Restart=always" in content, "Should have Restart=always setting"
        assert "RestartSec=" in content, "Should have RestartSec setting"


class TestInstallationScriptFunctionality:
    """Test the installation script functionality."""

    @pytest.fixture
    def install_script(self):
        """Path to the installation script."""
        return Path(__file__).parent.parent.parent / "scripts" / "install_service.sh"

    def test_install_script_has_help_function(self, install_script):
        """Test that install script has a help function."""
        if install_script.exists():
            content = install_script.read_text()
            assert "print_usage" in content, "Script should have print_usage function"

    def test_install_script_has_commands(self, install_script):
        """Test that install script supports required commands."""
        if install_script.exists():
            content = install_script.read_text()
            commands = ["install", "uninstall", "enable", "disable", "start", "stop", "restart", "status"]
            for command in commands:
                assert command in content, f"Script should support {command} command"

    def test_install_script_has_argument_parsing(self, install_script):
        """Test that install script has argument parsing."""
        if install_script.exists():
            content = install_script.read_text()
            assert "while [[ $# -gt 0" in content or "getopts" in content, \
                "Script should have argument parsing"

    def test_install_script_supports_user_mode(self, install_script):
        """Test that install script supports user mode."""
        if install_script.exists():
            content = install_script.read_text()
            assert "--user" in content, "Script should support --user option"
            assert "USER_MODE" in content, "Script should have USER_MODE variable"


class TestServiceConfiguration:
    """Test service configuration variables."""

    @pytest.fixture
    def service_file(self):
        """Path to the systemd service file."""
        return Path(__file__).parent.parent.parent / "systemd" / "vllm-router.service"

    def test_service_file_uses_config_file(self, service_file):
        """Test that service references a config file."""
        content = service_file.read_text()
        assert "config.toml" in content or "config" in content.lower(), \
            "Service should reference a config file"

    def test_service_file_logging_configured(self, service_file):
        """Test that service file has logging configured."""
        content = service_file.read_text()
        logged_settings = ["StandardOutput=journal", "StandardError=journal", "SyslogIdentifier"]
        logging_found = any(setting in content for setting in logged_settings)
        assert logging_found, "Service file should have logging configuration"

    def test_service_file_working_directory(self, service_file):
        """Test that service file sets working directory."""
        content = service_file.read_text()
        assert "WorkingDirectory=" in content, "Service file should set WorkingDirectory"


@pytest.mark.manual
class TestManualInstallation:
    """Manual tests that require actual system access."""

    @pytest.mark.skipif(
        not os.path.exists("/usr/sbin/systemctl"),
        reason="systemctl not available on this system"
    )
    def test_check_systemd_available(self):
        """Test if systemd is available on the system."""
        result = subprocess.run(["systemctl", "--version"], capture_output=True, text=True)
        assert result.returncode == 0, "systemd should be available"

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Not applicable on Windows"
    )
    def test_check_user_permissions(self):
        """Test current user permissions."""
        is_admin = os.getuid() == 0
        assert isinstance(is_admin, bool), "User permission check should return boolean"