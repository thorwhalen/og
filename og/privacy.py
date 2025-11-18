"""Privacy and security enhancements for OG.

This module provides enhanced privacy controls, data encryption,
and sensitive data protection.
"""

import hashlib
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Pattern

from og.base import Observation


class PrivacyControls:
    """Enhanced privacy features for OG.

    Provides:
    - Data encryption
    - Sensitive data redaction
    - Retention policies
    - Exclude patterns
    - Anonymization
    """

    def __init__(self):
        """Initialize privacy controls."""
        self.exclude_patterns: list[Pattern] = []
        self.exclude_urls: list[Pattern] = []
        self.exclude_apps: list[str] = []
        self.retention_days: Optional[int] = None
        self.encryption_enabled = False
        self._encryption_key: Optional[bytes] = None

        # Common sensitive patterns
        self._sensitive_patterns = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b\d{16}\b'),  # Credit card
            re.compile(r'\b[A-Za-z0-9]{32,}\b'),  # API keys (heuristic)
            re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'&<>\s]+)', re.I),
            re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'&<>\s]+)', re.I),
            re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'&<>\s]+)', re.I),
            re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'&<>\s]+)', re.I),
        ]

    def enable_encryption(self, password: Optional[str] = None) -> None:
        """Enable encryption for stored observations.

        Args:
            password: Optional password for encryption (uses system keyring if None)
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

            if password:
                # Derive key from password
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'og_salt',  # In production, use random salt
                    iterations=100000,
                )
                key = kdf.derive(password.encode())
            else:
                # Generate random key
                key = Fernet.generate_key()

                # Store in keyring
                try:
                    import keyring

                    keyring.set_password('og', 'encryption_key', key.decode())
                except Exception:
                    # Save to file if keyring not available
                    key_file = Path.home() / '.og' / 'encryption.key'
                    key_file.parent.mkdir(exist_ok=True, parents=True)
                    key_file.write_bytes(key)

            self._encryption_key = key
            self.encryption_enabled = True

        except ImportError:
            raise ImportError(
                "cryptography required for encryption. "
                "Install with: pip install cryptography"
            )

    def encrypt_data(self, data: str) -> str:
        """Encrypt data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data (base64 encoded)
        """
        if not self.encryption_enabled or not self._encryption_key:
            return data

        from cryptography.fernet import Fernet

        f = Fernet(self._encryption_key)
        encrypted = f.encrypt(data.encode())
        return encrypted.decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data.

        Args:
            encrypted_data: Encrypted data to decrypt

        Returns:
            Decrypted data
        """
        if not self.encryption_enabled or not self._encryption_key:
            return encrypted_data

        from cryptography.fernet import Fernet

        f = Fernet(self._encryption_key)
        decrypted = f.decrypt(encrypted_data.encode())
        return decrypted.decode()

    def redact_sensitive_data(self, text: str) -> str:
        """Automatically detect and redact sensitive data.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        redacted = text

        for pattern in self._sensitive_patterns:
            redacted = pattern.sub('[REDACTED]', redacted)

        return redacted

    def anonymize_observation(self, obs: Observation) -> Observation:
        """Anonymize an observation by removing personal data.

        Args:
            obs: Observation to anonymize

        Returns:
            Anonymized observation
        """
        # Create a copy
        from copy import deepcopy

        anon_obs = deepcopy(obs)

        # Redact sensitive data in all string fields
        for key, value in anon_obs.data.items():
            if isinstance(value, str):
                anon_obs.data[key] = self.redact_sensitive_data(value)

        # Hash any identifiers
        if 'user' in anon_obs.data:
            anon_obs.data['user'] = self._hash_identifier(anon_obs.data['user'])

        if 'email' in anon_obs.data:
            anon_obs.data['email'] = self._hash_identifier(anon_obs.data['email'])

        return anon_obs

    def add_exclude_pattern(self, pattern: str, pattern_type: str = 'general') -> None:
        """Add a pattern to exclude from observation.

        Args:
            pattern: Regex pattern to exclude
            pattern_type: Type of pattern ('general', 'url', 'app')
        """
        compiled = re.compile(pattern, re.IGNORECASE)

        if pattern_type == 'url':
            self.exclude_urls.append(compiled)
        elif pattern_type == 'app':
            self.exclude_apps.append(pattern)
        else:
            self.exclude_patterns.append(compiled)

    def should_exclude(self, obs: Observation) -> bool:
        """Check if an observation should be excluded based on patterns.

        Args:
            obs: Observation to check

        Returns:
            True if should be excluded
        """
        # Check URL exclusions
        if obs.event_type == 'browser_visit':
            url = obs.data.get('url', '')
            for pattern in self.exclude_urls:
                if pattern.search(url):
                    return True

        # Check app exclusions
        if obs.event_type in ['app_usage', 'app_switch']:
            app = obs.data.get('application', '')
            if app in self.exclude_apps:
                return True

        # Check general patterns
        obs_text = str(obs.data)
        for pattern in self.exclude_patterns:
            if pattern.search(obs_text):
                return True

        return False

    def set_retention_policy(self, days: int) -> None:
        """Set data retention policy.

        Args:
            days: Number of days to retain data
        """
        self.retention_days = days

    def apply_retention_policy(self, observations: list[Observation]) -> list[Observation]:
        """Apply retention policy to observations.

        Args:
            observations: Observations to filter

        Returns:
            Filtered observations (within retention period)
        """
        if self.retention_days is None:
            return observations

        cutoff = datetime.now() - timedelta(days=self.retention_days)

        return [obs for obs in observations if obs.timestamp >= cutoff]

    def generate_privacy_report(self, observations: list[Observation]) -> dict:
        """Generate a privacy report showing what data is being collected.

        Args:
            observations: Observations to analyze

        Returns:
            Privacy report
        """
        report = {
            'total_observations': len(observations),
            'observers': {},
            'event_types': {},
            'data_fields': set(),
            'potential_pii': [],
        }

        for obs in observations:
            # Count by observer
            observer = obs.observer_name
            if observer not in report['observers']:
                report['observers'][observer] = 0
            report['observers'][observer] += 1

            # Count by event type
            event_type = obs.event_type
            if event_type not in report['event_types']:
                report['event_types'][event_type] = 0
            report['event_types'][event_type] += 1

            # Track data fields
            for key in obs.data.keys():
                report['data_fields'].add(key)

            # Check for potential PII
            for key, value in obs.data.items():
                if isinstance(value, str):
                    for pattern in self._sensitive_patterns:
                        if pattern.search(value):
                            report['potential_pii'].append(
                                {
                                    'observer': observer,
                                    'event_type': event_type,
                                    'field': key,
                                }
                            )

        report['data_fields'] = list(report['data_fields'])

        return report

    def export_gdpr_data(self, observations: list[Observation], output_file: str) -> None:
        """Export data in GDPR-compliant format.

        Args:
            observations: Observations to export
            output_file: Path to output file
        """
        import json

        gdpr_data = {
            'export_date': datetime.now().isoformat(),
            'data_subject': 'User',
            'observations': [obs.to_dict() for obs in observations],
            'privacy_notice': 'This data was collected by Own Ghost (OG) for personal activity tracking.',
        }

        with open(output_file, 'w') as f:
            json.dump(gdpr_data, f, indent=2)

    def _hash_identifier(self, identifier: str) -> str:
        """Hash an identifier for anonymization.

        Args:
            identifier: Identifier to hash

        Returns:
            Hashed identifier
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def create_privacy_config() -> dict:
    """Create a default privacy configuration.

    Returns:
        Privacy configuration dictionary
    """
    return {
        'encryption_enabled': False,
        'redact_sensitive': True,
        'retention_days': 90,
        'exclude_patterns': [
            r'bank\.com',
            r'medical',
            r'health',
            r'password',
        ],
        'exclude_apps': [
            'Password Manager',
            'Bitwarden',
            '1Password',
            'LastPass',
        ],
    }


def apply_privacy_config(privacy_controls: PrivacyControls, config: dict) -> None:
    """Apply privacy configuration to controls.

    Args:
        privacy_controls: PrivacyControls instance
        config: Configuration dictionary
    """
    if config.get('encryption_enabled'):
        privacy_controls.enable_encryption()

    if config.get('retention_days'):
        privacy_controls.set_retention_policy(config['retention_days'])

    for pattern in config.get('exclude_patterns', []):
        privacy_controls.add_exclude_pattern(pattern, 'general')

    for app in config.get('exclude_apps', []):
        privacy_controls.add_exclude_pattern(app, 'app')
