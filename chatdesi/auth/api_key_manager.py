"""
Practical API key management with live testing.
"""

from typing import Dict, Optional, Iterable
import time

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class APIKeyManager:
    """API key management with live validation."""

    def __init__(self):
        # Use session state to cache test results to avoid re-testing keys
        if 'tested_keys' not in st.session_state:
            st.session_state.tested_keys = {'anthropic': {}, 'openai': {}, 'cborg_openai': {}, 'cborg_anthropic': {}}

    def test_openai_key(self, api_key: str) -> bool:
        """Test OpenAI API key with a minimal request."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False

    def test_cborg_openai_key(self, api_key: str) -> bool:
        """Test OpenAI API key with a minimal request."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.cborg.lbl.gov")
            client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Respond with 1"}],
                max_tokens=3
            )
            return True
        except Exception:
            return False

    def test_anthropic_key(self, api_key: str) -> bool:
        """Test Anthropic API key with a minimal request."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception:
            return False

    def test_cborg_anthropic_key(self, api_key: str) -> bool:
        """Test Anthropic API key with a minimal request."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, base_url="https://api.cborg.lbl.gov")
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception:
            return False

    def get_user_api_keys(self) -> Dict[str, str]:
        """Get and test API keys from user input."""
        # st.write("### 🔑 AI Model Configuration")
        api_keys = {}

        with st.sidebar.expander("Enter AI Model API Keys", expanded=True):
            openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            if openai_key:
                # Check for openai key, cache before making an API call
                if st.session_state.tested_keys['openai'].get(openai_key) or self.test_openai_key(openai_key):
                    api_keys["openai"] = openai_key
                    st.session_state.tested_keys['openai'][openai_key] = True
                    st.success("✅ OpenAI API key is valid.")
                else:
                    st.session_state.tested_keys['openai'][openai_key] = False
                    st.error("❌ Invalid OpenAI API key.")

            anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
            if anthropic_key:
                if st.session_state.tested_keys.get(anthropic_key) or self.test_anthropic_key(anthropic_key):
                    api_keys["anthropic"] = anthropic_key
                    st.session_state.tested_keys[anthropic_key] = True
                    st.success("✅ Anthropic API key is valid.")
                else:
                    st.error("❌ Invalid Anthropic API key.")

            cborg_key = st.text_input("cborg API Key", type="password", placeholder="sk-...")
            if cborg_key:
                # Check for cborg openai, cache before making an API call
                if st.session_state.tested_keys['cborg_openai'].get(cborg_key) or self.test_cborg_openai_key(cborg_key):
                    api_keys["cborg_openai"] = cborg_key
                    st.session_state.tested_keys['cborg_openai'][cborg_key] = True
                    st.success("✅ cborg API key is valid for openai.")
                else:
                    st.session_state.tested_keys['cborg_openai'][cborg_key] = False
                    st.error("❌ Invalid OpenAI API key.")
                # Check for cborg anthropic, cache before making an API call
                if st.session_state.tested_keys['cborg_anthropic'].get(cborg_key) or self.test_cborg_anthropic_key(cborg_key):
                    api_keys["cborg_anthropic"] = cborg_key
                    st.session_state.tested_keys['cborg_anthropic'][cborg_key] = True
                    st.success("✅ cborg API key is valid for anthropic.")
                else:
                    st.session_state.tested_keys['cborg_anthropic'][cborg_key] = False
                    st.error("❌ Invalid cborg Anthropic API key.")
        return api_keys

    def get_available_models(self, api_keys: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Get available models based on working API keys."""
        from ..config import settings
        available_models = {}
        all_models = settings.model.available_models

        if "openai" in api_keys or "cborg_openai" in api_keys:
            for k, v in all_models.items():
                if v["provider"] == "openai":
                    available_models[k] = v

        if "anthropic" in api_keys or "cborg_anthropic" in api_keys:
            for k, v in all_models.items():
                if v["provider"] == "anthropic":
                    available_models[k] = v

        return available_models

    def render_model_selector(self, available_models: Dict[str, Dict[str, str]]) -> Optional[str]:
        """Render model selection interface."""
        if not available_models:
            st.warning("⚠️ No working API keys found. Please enter valid API keys above.")
            return None

        # st.sidebar.write("### 🎯 Model Selection")

        model_options = {f"{info['display_name']} ({info['provider']})": key
                         for key, info in available_models.items()}

        selected_display = st.sidebar.selectbox("Choose AI Model:", options=list(model_options.keys()))
        return model_options[selected_display]


class SimpleModelClient:
    """Simple client wrapper that works with any provider."""

    def __init__(self, provider: str, api_key: str, model_name: str, base_url: str = None):
        self.base_url = base_url
        self.provider = provider
        self.model_name = model_name

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, base_url=self.base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _stream_completion(self, messages, max_tokens, temperature) -> Iterable[str]:
        """Private generator for handling streaming responses."""
        for char in f"{self.model_name}: ":
            yield char
            time.sleep(0.05)  # Simulate streaming delay
        if self.provider == "openai":
            stream = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        elif self.provider == "anthropic":
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_messages = [msg for msg in messages if msg["role"] != "system"]
            with self.client.messages.stream(
                model=self.model_name, max_tokens=max_tokens, temperature=temperature,
                messages=user_messages, system=system_message
            ) as stream:
                for text in stream.text_stream:
                    yield text

    def chat_completion(self, messages, max_tokens, temperature, stream=False):
        """Generate chat completion, supporting both streaming and non-streaming."""
        if stream:
            return self._stream_completion(messages, max_tokens, temperature)

        # Fallback for non-streaming requests
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return f"{self.model_name}: "+response.choices[0].message.content
        elif self.provider == "anthropic":
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_messages = [msg for msg in messages if msg["role"] != "system"]
            response = self.client.messages.create(
                model=self.model_name, max_tokens=max_tokens, temperature=temperature,
                messages=user_messages, system=system_message
            )
            return f"{self.model_name}: "+response.content[0].text

    def get_provider_name(self) -> str:
        return self.provider