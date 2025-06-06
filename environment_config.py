import os
import socket
from typing import Dict, Any, Optional

class EnvironmentConfig:
    """
    Environment configuration class for managing proxy settings, certificates, 
    and other environment-specific settings.
    """
    
    def __init__(self, environment: str = 'personal'):
        """
        Initialize environment configuration.
        
        Args:
            environment: Environment type ('corporate' or 'personal'). Defaults to 'personal'.
        """
        if environment not in ['corporate', 'personal']:
            raise ValueError("Environment must be 'corporate' or 'personal'")
            
        self.environment = environment
        self.is_corporate = self.environment == 'corporate'
        self.is_personal = self.environment == 'personal'
        
        # Configure environment
        self._configure_environment()
        
        print(f"Environment set to: {self.environment}")
        if self.is_corporate:
            print("Corporate proxy settings applied")
        else:
            print("Personal environment - no proxy configuration")

    def _configure_environment(self):
        """Configure environment settings based on specified environment."""
        if self.is_corporate:
            self._configure_corporate()
        else:
            self._configure_personal()
    
    def _configure_corporate(self):
        """Configure corporate environment settings."""
        # Set proxy environment variables
        os.environ['HTTP_PROXY'] = "http://application-proxy.blackrock.com:9443"
        os.environ['HTTPS_PROXY'] = "http://application-proxy.blackrock.com:9443"
        os.environ['NO_PROXY'] = "localhost,dev.blackrock.com,svn.blackrock.com,artifactory.blackrock.com,127.0.0.1,.bfm.com,.blackrock.com"
        
        # Certificate settings
        self.cert_path = "corp-proxy-ca.crt"
        self.verify_ssl = self.cert_path
        
        # Request settings
        self.trust_env = True
        self.timeout_settings = {
            'connect_timeout': 30,
            'read_timeout': 60,
            'total_timeout': 120
        }
        
        # Rate limiting (more conservative for corporate)
        self.rate_limit_settings = {
            'min_request_interval': 1.0,  # 1 second between requests
            'max_concurrent': 3,
            'retry_attempts': 5,
            'backoff_factor': 2.0
        }
    
    def _configure_personal(self):
        """Configure personal environment settings."""
        # Clear any existing proxy settings
        for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            if proxy_var in os.environ:
                del os.environ[proxy_var]
        
        # Certificate settings
        self.cert_path = None
        self.verify_ssl = True  # Use default SSL verification
        
        # Request settings
        self.trust_env = False
        self.timeout_settings = {
            'connect_timeout': 10,
            'read_timeout': 30,
            'total_timeout': 60
        }
        
        # Rate limiting (more aggressive for personal)
        self.rate_limit_settings = {
            'min_request_interval': 0.5,  # 500ms between requests
            'max_concurrent': 10,
            'retry_attempts': 3,
            'backoff_factor': 1.5
        }
    
    def get_requests_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for requests library calls.
        
        Returns:
            Dictionary of kwargs for requests.get/post calls
        """
        kwargs = {
            'timeout': (self.timeout_settings['connect_timeout'], 
                       self.timeout_settings['read_timeout']),
        }
        
        if self.is_corporate:
            kwargs['verify'] = self.cert_path
        else:
            kwargs['verify'] = self.verify_ssl
        
        return kwargs
    
    def get_aiohttp_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for aiohttp ClientSession.
        
        Returns:
            Dictionary of kwargs for aiohttp.ClientSession
        """
        import aiohttp
        
        timeout = aiohttp.ClientTimeout(
            connect=self.timeout_settings['connect_timeout'],
            total=self.timeout_settings['total_timeout']
        )
        
        kwargs = {
            'timeout': timeout,
            'trust_env': self.trust_env
        }
        
        if self.is_corporate:
            # For corporate environment with custom cert
            import ssl
            ssl_context = ssl.create_default_context()
            if os.path.exists(self.cert_path):
                ssl_context.load_verify_locations(self.cert_path)
            kwargs['connector'] = aiohttp.TCPConnector(ssl=ssl_context)
        
        return kwargs
    
    def print_config(self):
        """Print current configuration for debugging."""
        print(f"\n{'='*50}")
        print(f"ENVIRONMENT CONFIGURATION")
        print(f"{'='*50}")
        print(f"Environment: {self.environment}")
        print(f"Certificate path: {self.cert_path}")
        print(f"SSL verification: {self.verify_ssl}")
        print(f"Trust environment: {self.trust_env}")
        print(f"Proxy settings:")
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']:
            value = os.environ.get(key, 'Not set')
            print(f"  {key}: {value}")
        print(f"Rate limiting: {self.rate_limit_settings}")
        print(f"Timeouts: {self.timeout_settings}")
        print(f"{'='*50}")

# Global configuration instance - defaults to 'personal'
# You can override by setting ENV_CONFIG = EnvironmentConfig('corporate')
ENV_CONFIG = EnvironmentConfig()

# Convenience functions
def is_corporate_environment() -> bool:
    """Check if running in corporate environment."""
    return ENV_CONFIG.is_corporate

def is_personal_environment() -> bool:
    """Check if running in personal environment."""
    return ENV_CONFIG.is_personal

def get_requests_kwargs() -> Dict[str, Any]:
    """Get requests library kwargs for current environment."""
    return ENV_CONFIG.get_requests_kwargs()

def get_aiohttp_kwargs() -> Dict[str, Any]:
    """Get aiohttp kwargs for current environment."""
    return ENV_CONFIG.get_aiohttp_kwargs()

def set_environment(env_type: str):
    """Set specific environment configuration."""
    global ENV_CONFIG
    ENV_CONFIG = EnvironmentConfig(env_type)
    print(f"Environment set to: {env_type}")

# Example usage and testing
if __name__ == "__main__":
    # Test default environment (personal)
    config = EnvironmentConfig()
    config.print_config()
    
    # Test setting different environments
    print("\nTesting personal environment:")
    personal_config = EnvironmentConfig('personal')
    personal_config.print_config()
    
    print("\nTesting corporate environment:")
    corporate_config = EnvironmentConfig('corporate')
    corporate_config.print_config()