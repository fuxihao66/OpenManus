import asyncio
import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)
    
    # Persistent state settings
    cache_dir: str = Field(default="./cache", exclude=True)
    user_data_dir: Optional[str] = Field(default=None, exclude=True)
    playwright: Optional[Any] = Field(default=None, exclude=True)
    browser_type: Optional[Any] = Field(default=None, exclude=True)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _human_like_delay(self, min_delay=0.1, max_delay=0.5):
        """Add a random human-like delay"""
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)

    async def _human_like_mouse_movement(self, page, target_element=None):
        """Simulate human-like mouse movement"""
        if target_element:
            # Get element position
            box = await target_element.bounding_box()
            if box:
                # Move mouse to element with some randomness
                target_x = box['x'] + box['width'] / 2 + random.randint(-10, 10)
                target_y = box['y'] + box['height'] / 2 + random.randint(-10, 10)
                
                # Move mouse with intermediate steps
                current_x = random.randint(0, 1280)
                current_y = random.randint(0, 720)
                
                steps = random.randint(3, 8)
                for i in range(steps):
                    intermediate_x = current_x + (target_x - current_x) * (i + 1) / steps
                    intermediate_y = current_y + (target_y - current_y) * (i + 1) / steps
                    await page.mouse.move(intermediate_x, intermediate_y)
                    await self._human_like_delay(0.01, 0.05)
        
        # Random mouse movement to simulate human behavior
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, 1280)
            y = random.randint(0, 720)
            await page.mouse.move(x, y)
            await self._human_like_delay(0.01, 0.03)

    async def _simulate_human_behavior(self, page):
        """Simulate human-like behavior patterns"""
        # Random viewport scrolling
        if random.random() < 0.3:  # 30% chance
            scroll_amount = random.randint(100, 500)
            await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            await self._human_like_delay(0.5, 1.5)
        
        # Random mouse movements
        if random.random() < 0.4:  # 40% chance
            await self._human_like_mouse_movement(page)
        
        # Random tab switching simulation
        if random.random() < 0.1:  # 10% chance
            await page.bring_to_front()
            await self._human_like_delay(0.2, 0.8)
        
        # Random focus changes
        if random.random() < 0.2:  # 20% chance
            await page.evaluate("document.body.focus()")
            await self._human_like_delay(0.1, 0.5)

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized with persistent state."""
        if self.browser is None:
            try:
                print("Initializing browser...")
                
                # Create cache directory if it doesn't exist
                cache_path = Path(self.cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                
                # Set up persistent user data directory
                self.user_data_dir = str(cache_path / "browser_profile")
                user_data_path = Path(self.user_data_dir)
                user_data_path.mkdir(parents=True, exist_ok=True)

                # Use Playwright's browser type directly to create persistent context
                from playwright.async_api import async_playwright
                
                print("Starting Playwright...")
                self.playwright = await async_playwright().start()
                self.browser_type = self.playwright.chromium
                
                # Simplified browser configuration for debugging
                print("Launching browser context...")
                persistent_context = await self.browser_type.launch_persistent_context(
                    user_data_dir=self.user_data_dir,
                    headless=False,  # Ensure browser is visible
                    accept_downloads=True,
                    viewport={"width": 1920, "height": 1080},
                    # Basic anti-detection arguments
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-web-security',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-extensions',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-breakpad',
                        '--disable-client-side-phishing-detection',
                        '--disable-default-apps',
                        '--disable-hang-monitor',
                        '--disable-ipc-flooding-protection',
                        '--disable-popup-blocking',
                        '--disable-prompt-on-repost',
                        '--disable-renderer-backgrounding',
                        '--disable-sync',
                        '--disable-translate',
                        '--metrics-recording-only',
                        '--no-first-run',
                        '--no-default-browser-check',
                        '--password-store=basic',
                        '--use-mock-keychain',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-notifications',
                        '--window-position=0,0',
                        '--window-size=1920,1080',
                    ],
                    # Set a realistic user agent
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                    # Basic settings
                    locale='en-US',
                    timezone_id='America/New_York',
                    java_script_enabled=True,
                    bypass_csp=True,
                    ignore_https_errors=False,
                    color_scheme='light',
                    geolocation={"latitude": 40.7128, "longitude": -74.0060},
                    # Basic permissions
                    permissions=["geolocation", "notifications", "camera", "microphone"],
                )
                
                print("Browser context launched successfully!")
                
            except Exception as e:
                print(f"Error launching browser: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Create a complete browser wrapper that implements all required methods
            class PersistentBrowserWrapper:
                def __init__(self, persistent_context, browser_type, playwright):
                    self._persistent_context = persistent_context
                    self.browser_type = browser_type
                    self.playwright = playwright
                    self.config = BrowserConfig()
                    self.contexts = [persistent_context]
                
                async def get_playwright_browser(self):
                    return self
                
                async def new_context(self, **kwargs):
                    # Return the persistent context instead of creating a new one
                    return self._persistent_context
                
                async def close(self):
                    await self._persistent_context.close()
                    await self.playwright.stop()
            
            print("Creating browser wrapper...")
            # Create wrapper browser
            self.browser = PersistentBrowserWrapper(persistent_context, self.browser_type, self.playwright)
            
            print("Creating browser context...")
            # Create a BrowserContext instance with the persistent context
            context_config = BrowserContextConfig()
            
            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config
            
            # Manually create the context with the persistent Playwright context
            from browser_use.browser.context import BrowserContext
            self.context = BrowserContext(self.browser, context_config)
            
            # Replace the internal context with the persistent one and set it as already created
            self.context._context = persistent_context
            self.context._is_initialized = True
            
            print("Getting browser page...")
            # Initialize DOM service with the existing page or create a new one
            try:
                # Try to get the first page if it exists
                pages = persistent_context.pages
                if pages:
                    page = pages[0]
                    print("Using existing page")
                else:
                    page = await persistent_context.new_page()
                    print("Created new page")
            except Exception as e:
                print(f"Error getting page: {e}")
                page = await persistent_context.new_page()
                print("Created new page after error")
            
            print("Adding stealth script...")
            # Add basic stealth JavaScript
            stealth_script = """
                // Basic stealth script
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                    configurable: true
                });
                
                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                    configurable: true
                });
                
                // Mock platform
                Object.defineProperty(navigator, 'platform', {
                    get: () => 'Win32',
                    configurable: true
                });
                
                // Basic Chrome mock
                window.chrome = {
                    runtime: {
                        id: 'abcdefghijklmnopqrstuvwxyzabcdef',
                        getURL: function(path) { return 'chrome-extension://' + this.id + '/' + path; }
                    }
                };
                
                // Remove automation flags
                delete window._phantom;
                delete window.callPhantom;
                delete window.phantom;
            """
            
            await page.add_init_script(stealth_script)
            
            # Also add the script to the context so it applies to all new pages
            await persistent_context.add_init_script(stealth_script)
            
            # Set basic headers
            headers = {
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
            }
            
            await page.set_extra_http_headers(headers)
            
            # Add event listener to set headers on all new pages
            async def set_headers_on_new_page(new_page):
                await new_page.set_extra_http_headers(headers)
            
            persistent_context.on("page", lambda new_page: asyncio.create_task(set_headers_on_new_page(new_page)))
            
            print("Creating DOM service...")
            self.dom_service = DomService(page)
            
            print("Browser initialization completed successfully!")

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    
                    # Add human-like behavior
                    await self._human_like_delay(0.5, 1.5)
                    await self._human_like_mouse_movement(page)
                    
                    # Apply domain-specific bypass
                    if 'google.com' in url or 'accounts.google.com' in url:
                        await self.google_login_bypass(page)
                    elif 'discord.com' in url:
                        await self.discord_bypass(page)
                    
                    await page.goto(url)
                    await page.wait_for_load_state()
                    
                    # Add human behavior after navigation
                    await self._simulate_human_behavior(page)
                    
                    # Add some random delay after navigation
                    await self._human_like_delay(1.0, 2.0)
                    
                    return ToolResult(output=f"Navigated to {url}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)
                    await page.wait_for_load_state()

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    
                    page = await context.get_current_page()
                    
                    # Add human-like behavior before clicking
                    await self._human_like_delay(0.3, 0.8)
                    
                    # Get the actual element from the page
                    try:
                        target_element = await page.query_selector(element.xpath)
                        if target_element:
                            await self._human_like_mouse_movement(page, target_element)
                            await self._human_like_delay(0.1, 0.3)
                    except:
                        pass
                    
                    download_path = await context._click_element_node(element)
                    
                    # Add delay after clicking
                    await self._human_like_delay(0.5, 1.0)
                    
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()
                    import markdownify

                    content = markdownify.markdownify(await page.content())

                    prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                    messages = [{"role": "system", "content": prompt}]

                    # Define extraction function schema
                    extraction_function = {
                        "type": "function",
                        "function": {
                            "name": "extract_content",
                            "description": "Extract specific information from a webpage based on a goal",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "extracted_content": {
                                        "type": "object",
                                        "description": "The content extracted from the page according to the goal",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text content extracted from the page",
                                            },
                                            "metadata": {
                                                "type": "object",
                                                "description": "Additional metadata about the extracted content",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "Source of the extracted content",
                                                    }
                                                },
                                            },
                                        },
                                    }
                                },
                                "required": ["extracted_content"],
                            },
                        },
                    }

                    # Use LLM to extract content with required function calling
                    response = await self.llm.ask_tool(
                        messages,
                        tools=[extraction_function],
                        tool_choice="required",
                    )

                    if response and response.tool_calls:
                        args = json.loads(response.tool_calls[0].function.arguments)
                        extracted_content = args.get("extracted_content", {})
                        return ToolResult(
                            output=f"Extracted from page:\n{extracted_content}\n"
                        )

                    return ToolResult(output="No content was extracted from the page.")

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def google_login_bypass(self, page):
        """
        Additional measures specifically for Google login bypass.
        This should be called before attempting Google login.
        """
        try:
            # Additional JavaScript injection for Google-specific bypass
            google_bypass_script = """
                // Additional Google-specific stealth measures
                
                // Override navigator properties that Google specifically checks
                Object.defineProperty(navigator, 'vendor', {
                    get: () => 'Google Inc.',
                    configurable: true
                });
                
                Object.defineProperty(navigator, 'productSub', {
                    get: () => '20030107',
                    configurable: true
                });
                
                // Mock specific Chrome properties that Google verifies
                if (window.chrome) {
                    window.chrome.app = {
                        isInstalled: false,
                        InstallState: 'disabled',
                        RunningState: 'cannot_run'
                    };
                    
                    window.chrome.csi = function() {
                        return {
                            onloadT: Date.now(),
                            pageT: Date.now() - 1000,
                            startE: Date.now() - 1500,
                            tran: 15
                        };
                    };
                }
                
                // Remove any automation痕迹 from localStorage
                try {
                    localStorage.removeItem('_puppeteer_');
                    localStorage.removeItem('_selenium');
                    localStorage.removeItem('_playwright');
                } catch (e) {}
                
                // Add Google-specific user behavior simulation
                let googleInteractionCount = 0;
                document.addEventListener('click', () => {
                    googleInteractionCount++;
                });
                
                // Override performance.memory if available
                if (performance.memory) {
                    Object.defineProperty(performance, 'memory', {
                        get: () => ({
                            usedJSHeapSize: 10000000,
                            totalJSHeapSize: 20000000,
                            jsHeapSizeLimit: 4000000000
                        }),
                        configurable: true
                    });
                }
            """
            
            await page.add_init_script(google_bypass_script)
            
            # Set referrer policy to look more natural
            await page.evaluate("""
                Object.defineProperty(document, 'referrer', {
                    get: () => 'https://www.google.com/',
                    configurable: true
                });
            """)
            
            return True
        except Exception as e:
            print(f"Google login bypass failed: {e}")
            return False

    async def discord_bypass(self, page):
        """
        Additional measures specifically for Discord to prevent BigInt errors.
        This should be called before navigating to Discord.
        """
        try:
            # Discord-specific JavaScript injection to prevent detection and errors
            discord_bypass_script = """
                // Discord-specific stealth and error fixes - COMPREHENSIVE VERSION
                
                // Comprehensive anti-detection for Discord
                if (typeof navigator !== 'undefined') {
                    // Hide webdriver flag
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                        configurable: true
                    });
                    
                    // Hide automation flags
                    Object.defineProperty(navigator, 'permissions', {
                        get: () => ({
                            query: () => Promise.resolve({ state: 'granted' })
                        }),
                        configurable: true
                    });
                    
                    // Mock plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [
                            {
                                0: {type: "application/x-google-chrome-pdf"},
                                description: "Portable Document Format",
                                filename: "internal-pdf-viewer",
                                length: 1,
                                name: "Chrome PDF Plugin"
                            },
                            {
                                0: {type: "application/x-nacl"},
                                description: "Native Client",
                                filename: "internal-nacl-plugin",
                                length: 1,
                                name: "Native Client"
                            }
                        ],
                        configurable: true
                    });
                    
                    // Mock languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en', 'zh-CN', 'zh'],
                        configurable: true
                    });
                    
                    // Mock platform
                    Object.defineProperty(navigator, 'platform', {
                        get: () => 'Win32',
                        configurable: true
                    });
                    
                    // Mock hardware concurrency
                    Object.defineProperty(navigator, 'hardwareConcurrency', {
                        get: () => 8,
                        configurable: true
                    });
                }
                
                // Mock Chrome API for Discord
                if (typeof window !== 'undefined') {
                    window.chrome = {
                        runtime: {
                            id: 'abcdefghijklmnopqrstuvwxyzabcdef',
                            getURL: function(path) { return 'chrome-extension://' + this.id + '/' + path; },
                            getManifest: function() { return { version: '1.0.0' }; },
                            connect: function() { return { postMessage: function() {}, onMessage: { addListener: function() {} } }; },
                            sendMessage: function() { return Promise.resolve(); },
                            onMessage: { addListener: function() {} },
                            onConnect: { addListener: function() {} }
                        },
                        storage: {
                            local: {
                                get: function() { return Promise.resolve({}); },
                                set: function() { return Promise.resolve(); }
                            },
                            sync: {
                                get: function() { return Promise.resolve({}); },
                                set: function() { return Promise.resolve(); }
                            }
                        },
                        tabs: {
                            query: function() { return Promise.resolve([]); },
                            create: function() { return Promise.resolve({ id: 1 }); },
                            update: function() { return Promise.resolve(); }
                        }
                    };
                    
                    // Mock Discord-specific APIs
                    window.discord = window.discord || {};
                    window.discord.experiments = window.discord.experiments || [];
                    
                    // Remove automation traces
                    delete window._phantom;
                    delete window.callPhantom;
                    delete window.phantom;
                    delete window.__nightmare;
                    delete window._Selenium_IDE_Recorder;
                }
                
                // Comprehensive error handling for Discord
                if (typeof window !== 'undefined') {
                    window.addEventListener('error', function(e) {
                        if (e.message && (
                            e.message.includes('cannot be converted to a BigInt') ||
                            e.message.includes('addEventListener is not a function') ||
                            e.message.includes('Maximum call stack size exceeded') ||
                            e.message.includes('JavaScript is disabled') ||
                            e.message.includes('enable JavaScript')
                        )) {
                            e.preventDefault();
                            e.stopPropagation();
                            return false;
                        }
                    }, true);
                    
                    // Handle unhandled promise rejections
                    window.addEventListener('unhandledrejection', function(e) {
                        if (e.reason && (
                            (e.reason.message && (
                                e.reason.message.includes('BigInt') ||
                                e.reason.message.includes('addEventListener') ||
                                e.reason.message.includes('Maximum call stack size exceeded') ||
                                e.reason.message.includes('JavaScript') ||
                                e.reason.message.includes('enable')
                            )) ||
                            (typeof e.reason === 'string' && (
                                e.reason.includes('BigInt') ||
                                e.reason.includes('addEventListener') ||
                                e.reason.includes('JavaScript')
                            ))
                        )) {
                            e.preventDefault();
                            return false;
                        }
                    }, true);
                }
                
                // Ensure JavaScript is enabled for Discord
                if (typeof document !== 'undefined') {
                    // Remove any noscript elements that might be showing the JavaScript disabled message
                    const noscripts = document.getElementsByTagName('noscript');
                    for (let i = noscripts.length - 1; i >= 0; i--) {
                        noscripts[i].parentNode.removeChild(noscripts[i]);
                    }
                }
            """
            
            await page.add_init_script(discord_bypass_script)
            
            # Set viewport to Discord's preferred dimensions
            await page.set_viewport_size({"width": 1280, "height": 720})
            
            # Wait a bit before navigation
            await page.wait_for_timeout(1000)
            
            # Set extra headers specifically for Discord
            await page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
            })
            
            return True
        except Exception as e:
            print(f"Discord bypass failed: {e}")
            return False

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources and save persistent state."""
        async with self.lock:
            if self.context is not None:
                # Persistent context automatically saves state, no need to manually save
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None
                self.playwright = None
                self.browser_type = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
