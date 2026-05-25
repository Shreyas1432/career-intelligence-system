import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

from src.core.browser.manager import BrowserManager
from src.core.config.browser import BrowserConfig

logger = logging.getLogger("src.modules.automation.browser")

# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class BrowserAssistanceError(RuntimeError):
    """
    Base exception for browser automation assistance errors.
    """


class SafetyGuardViolationError(BrowserAssistanceError):
    """
    Raised when safety boundaries (e.g. spam, bulk actions, or autonomous execution) are violated.
    """


class WorkflowStepFailedError(BrowserAssistanceError):
    """
    Raised when executing a workflow step fails.
    """


class UserAbortedWorkflowError(BrowserAssistanceError):
    """
    Raised when the user rejects a required execution step or aborts.
    """


# ------------------------------------------------------------------------------
# Config & Schemas
# ------------------------------------------------------------------------------

class BrowserAssistanceConfig(BaseModel):
    """
    Configuration settings for controlled browser automation.
    """

    headless: bool = Field(
        default=False,
        description="Whether to run the browser headlessly. Default is False for human-in-the-loop visibility.",
    )
    typing_speed_ms: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Keystroke interval in milliseconds to simulate human typing.",
    )
    step_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description="Enforced delay between steps to mimic human pacing.",
    )
    max_messaging_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum messages allowed per browser session to prevent mass spamming.",
    )


class ActionStep(BaseModel):
    """
    Represents a single step in a browser automation workflow.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(description="Unique identifier for the action step")
    action_type: Literal["navigate", "fill", "upload", "click", "wait_user"] = Field(
        description="Type of Playwright action to perform"
    )
    description: str = Field(description="User-friendly explanation of what this step does")
    target_selector: str | None = Field(
        default=None, description="CSS selector for the target element (if applicable)"
    )
    value: Any | None = Field(default=None, description="Payload value (e.g. text to write, path)")
    status: Literal["pending", "approved", "rejected", "completed", "failed"] = Field(
        default="pending", description="Current execution state of the step"
    )
    error_message: str | None = Field(
        default=None, description="Captured exception details if step execution fails"
    )

    def approve(self) -> None:
        """Marks the step as approved by the user."""
        self.status = "approved"

    def reject(self) -> None:
        """Marks the step as rejected by the user."""
        self.status = "rejected"


class Workflow(BaseModel):
    """
    A sequential container for browser assistance steps.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique workflow run ID")
    steps: list[ActionStep] = Field(default_factory=list, description="Sequence of actions")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Metadata tags for execution tracking"
    )


class FormMapping(BaseModel):
    """
    Model representing parsed candidate details for form-filling.
    """

    model_config = ConfigDict(extra="ignore")

    email: str | None = Field(default=None, description="Candidate email address")
    first_name: str | None = Field(default=None, description="Candidate first name")
    last_name: str | None = Field(default=None, description="Candidate last name")
    full_name: str | None = Field(default=None, description="Candidate full name")
    phone: str | None = Field(default=None, description="Candidate phone number")
    linkedin_url: str | None = Field(default=None, description="Candidate LinkedIn profile link")
    github_url: str | None = Field(default=None, description="Candidate GitHub profile link")
    portfolio_url: str | None = Field(default=None, description="Candidate portfolio website link")
    cover_letter: str | None = Field(default=None, description="Custom outreach draft or bio")
    additional_fields: dict[str, str] = Field(
        default_factory=dict, description="Extra form fields mapped as key-value pairs"
    )


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

async def highlight_element(page: Page, selector: str) -> bool:
    """
    Highlights a DOM element with a colored outline and scrolls it into view.
    Returns True if successfully highlighted, False if element is not found.
    """
    try:
        element = page.locator(selector).first
        count = await element.count()
        if count == 0:
            return False

        # Injects script to set element border/outline styling
        await page.evaluate(
            """
            (sel) => {
                const el = document.querySelector(sel);
                if (el) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    const originalOutline = el.style.outline;
                    const originalTransition = el.style.transition;
                    el.style.transition = 'outline 0.15s ease-in-out';
                    el.style.outline = '3px solid #ff4b4b';
                    el.style.outlineOffset = '2px';

                    // Flash outline to grab attention
                    setTimeout(() => {
                        el.style.outline = '3px solid #00c853';
                    }, 500);
                    setTimeout(() => {
                        el.style.outline = '3px solid #ff9100';
                    }, 1000);
                }
            }
            """,
            selector,
        )
        return True
    except Exception:
        return False


async def detect_fields(page: Page) -> list[dict[str, Any]]:
    """
    Scans the current page DOM to find inputs, textareas, and select elements,
    extracting metadata (id, name, type, placeholders, and associated labels).
    """
    try:
        raw_fields = await page.evaluate("""
            () => {
                const results = [];
                const selector = "input:not([type='hidden']):not([type='submit']):not([type='checkbox']):not([type='radio']), textarea, select";
                const elements = document.querySelectorAll(selector);

                elements.forEach((el, index) => {
                    // Skip invisible elements
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' || el.offsetWidth === 0) {
                        return;
                    }

                    // Get label text
                    let labelText = "";
                    if (el.id) {
                        const label = document.querySelector(`label[for="${el.id}"]`);
                        if (label) labelText = label.innerText;
                    }
                    if (!labelText) {
                        const parentLabel = el.closest("label");
                        if (parentLabel) labelText = parentLabel.innerText;
                    }

                    results.push({
                        index: index,
                        id: el.id || "",
                        name: el.name || "",
                        type: el.type || "",
                        placeholder: el.placeholder || "",
                        ariaLabel: el.getAttribute("aria-label") || "",
                        labelText: labelText || "",
                        tagName: el.tagName.toLowerCase()
                    });
                });
                return results;
            }
            """)
        return list(raw_fields)
    except Exception:
        return []


def classify_field(field_meta: dict[str, Any]) -> str | None:
    """
    Classifies a DOM field into a standard form mapping category
    based on label text, name, ID, and placeholder text.
    """
    id_val = str(field_meta.get("id", "")).lower()
    name_val = str(field_meta.get("name", "")).lower()
    placeholder_val = str(field_meta.get("placeholder", "")).lower()
    aria_label = str(field_meta.get("ariaLabel", "")).lower()
    label_text = str(field_meta.get("labelText", "")).lower()
    type_val = str(field_meta.get("type", "")).lower()

    combined_text = f"{id_val} {name_val} {placeholder_val} {aria_label} {label_text}"

    # 1. Resume File
    if type_val == "file" or any(x in combined_text for x in ["resume", "cv", "curriculum"]):
        return "resume_file"

    # Define keyword lists for matching
    keywords: list[tuple[str, str | list[str]]] = [
        ("email", "email"),
        ("phone", ["phone", "tel", "mobile", "contact number"]),
        ("linkedin_url", "linkedin"),
        ("github_url", "github"),
        ("portfolio_url", ["portfolio", "website", "personal site", "homepage"]),
        ("first_name", ["first name", "firstname"]),
        ("last_name", ["last name", "lastname"]),
        ("full_name", "name"),
    ]

    for category, pattern in keywords:
        if isinstance(pattern, list):
            if any(x in combined_text for x in pattern):
                # Avoid collision between first/last/full name
                if category == "first_name" and "last" in combined_text:
                    continue
                if category == "last_name" and "first" in combined_text:
                    continue
                return category
        elif pattern in combined_text:
            if category == "full_name" and ("first" in combined_text or "last" in combined_text):
                continue
            return category

    # 2. Cover Letter / Custom Notes
    if field_meta.get("tagName") == "textarea" or any(
        x in combined_text for x in ["cover letter", "message to hiring manager", "notes", "bio"]
    ):
        return "cover_letter"

    return None


def generate_css_selector(field_meta: dict[str, Any]) -> str:
    """
    Creates a unique CSS selector for an input field.
    """
    id_val = field_meta.get("id")
    if id_val:
        return f"#{id_val}"

    name_val = field_meta.get("name")
    tag_name = field_meta.get("tagName", "input")
    if name_val:
        return f"{tag_name}[name='{name_val}']"

    type_val = field_meta.get("type")
    placeholder_val = field_meta.get("placeholder")
    if type_val and placeholder_val:
        return f"{tag_name}[type='{type_val}'][placeholder='{placeholder_val}']"

    # Fallback to index-based selector
    index = field_meta.get("index", 0)
    return f"{tag_name}:nth-of-type({index + 1})"


def is_linkedin_url(url: str) -> bool:
    """
    Validates whether the provided URL belongs to LinkedIn.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return "linkedin.com" in domain
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------------------

class BrowserAssistanceEngine:
    """
    Orchestrates user-guided, human-in-the-loop browser interactions.
    Provides automation assistance for LinkedIn and job boards using headful Playwright.
    """

    def __init__(self, config: BrowserAssistanceConfig | None = None) -> None:
        self.config = config or BrowserAssistanceConfig()
        self._manager: BrowserManager | None = None
        self._is_running = False

        # Anti-spam message tracking counts within the active session
        self._message_count = 0

    async def start(self) -> None:
        """
        Starts the underlying headful Playwright browser manager.
        """
        if self._is_running:
            return

        logger.info("Starting headful browser automation assistance layer...")
        manager_config = BrowserConfig(
            headless=self.config.headless,
            max_browser_instances=1,
            max_contexts=1,
            navigation_timeout_ms=20_000,
            action_timeout_ms=10_000,
        )
        self._manager = BrowserManager(config=manager_config)
        await self._manager.start()
        self._is_running = True
        self._message_count = 0

    async def close(self) -> None:
        """
        Closes page, contexts, and Playwright launcher cleanly.
        """
        if self._manager:
            logger.info("Cleaning up browser assistance resources...")
            await self._manager.close()
            self._manager = None
        self._is_running = False

    async def __aenter__(self) -> "BrowserAssistanceEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def create_linkedin_workflow(
        self, username: str, section: Literal["profile", "experience", "skills", "about"]
    ) -> Workflow:
        """
        Generates a workflow to assist with navigating to specific LinkedIn sections.
        """
        workflow_id = f"linkedin-{section}-{uuid4().hex[:8]}"

        # Construct target URL
        if section == "profile":
            target_url = f"https://www.linkedin.com/in/{username}/"
        else:
            target_url = f"https://www.linkedin.com/in/{username}/details/{section}/"

        steps = [
            ActionStep(
                id="check-login",
                action_type="wait_user",
                description="Please log in to your LinkedIn account in the headful browser window.",
            ),
            ActionStep(
                id="navigate-section",
                action_type="navigate",
                description=f"Navigate to LinkedIn {section} section: {target_url}",
                value=target_url,
            ),
        ]
        return Workflow(
            id=workflow_id,
            steps=steps,
            metadata={"type": "linkedin_navigation", "section": section, "username": username},
        )

    async def create_job_application_workflow(
        self, url: str, form_data: FormMapping, resume_path: str | None = None
    ) -> Workflow:
        """
        Scans a job application page, detects fields, and maps them to a series of proposed ActionSteps.
        Important: Excludes automated submit buttons to ensure final sign-off is human.
        """
        if not self._is_running or not self._manager:
            raise BrowserAssistanceError("Browser assistance engine is not started.")

        workflow_id = f"job-app-{uuid4().hex[:8]}"
        steps = []

        # 1. Propose navigation to the job page
        steps.append(
            ActionStep(
                id="navigate-job",
                action_type="navigate",
                description=f"Navigate to job application page: {url}",
                value=url,
            )
        )

        # To build form-fill steps, we need to inspect the page DOM.
        # We start the browser context and load the page to scan it.
        async with self._manager.page() as page:
            logger.info(f"Scanning target job application page: {url}")
            await page.goto(url, wait_until="domcontentloaded")

            # Detect input fields
            detected = await detect_fields(page)
            for field in detected:
                category = classify_field(field)
                if not category:
                    continue

                selector = generate_css_selector(field)

                # Formulate steps based on classification
                if category == "resume_file":
                    if resume_path:
                        steps.append(
                            ActionStep(
                                id=f"upload-{field.get('id', 'resume')}",
                                action_type="upload",
                                description=f"Upload resume file ({resume_path}) to the field: '{field.get('labelText')}'",
                                target_selector=selector,
                                value=resume_path,
                            )
                        )
                else:
                    value = getattr(form_data, category, None)
                    if not value and category == "cover_letter":
                        value = form_data.cover_letter
                    if not value and field.get("name") in form_data.additional_fields:
                        value = form_data.additional_fields[field["name"]]

                    if value:
                        # Safeguard against spamming cover letters/outreaches
                        if category == "cover_letter":
                            self._check_message_limits()

                        steps.append(
                            ActionStep(
                                id=f"fill-{field.get('id') or field.get('name')}",
                                action_type="fill",
                                description=f"Fill field '{field.get('labelText') or field.get('name')}' with: '{value}'",
                                target_selector=selector,
                                value=value,
                            )
                        )

            # Inform user to manually submit
            steps.append(
                ActionStep(
                    id="user-review-submit",
                    action_type="wait_user",
                    description="Please review all filled fields and manually click the submit button in the browser.",
                )
            )

        return Workflow(
            id=workflow_id, steps=steps, metadata={"type": "job_application", "url": url}
        )

    async def run_workflow(self, workflow: Workflow, page: Page) -> AsyncIterator[ActionStep]:
        """
        Executes a workflow yielding each ActionStep one-by-one to the caller for confirmation.
        A step will only be executed in Playwright if its status is marked 'approved' (by the caller/user).
        """
        logger.info(f"Running workflow: {workflow.id} ({len(workflow.steps)} steps)")

        for step in workflow.steps:
            if step.status == "completed":
                continue

            # Yield step to allow caller to inspect and approve/reject
            yield step

            if step.status != "approved":
                logger.warning(f"Step {step.id} was not approved. Status: {step.status}. Skipping.")
                if step.status == "pending":
                    step.status = "rejected"
                continue

            # Execute approved step
            try:
                await self._execute_single_step(page, step)
                step.status = "completed"
                # Natural pacing delay to mimic human speed and minimize CPU
                await asyncio.sleep(self.config.step_delay_seconds)
            except Exception as exc:
                step.status = "failed"
                step.error_message = str(exc)
                logger.error(f"Step {step.id} failed: {exc}", exc_info=True)
                raise WorkflowStepFailedError(
                    f"Workflow execution failed at step {step.id}: {exc}"
                ) from exc

    async def _execute_single_step(self, page: Page, step: ActionStep) -> None:
        """
        Executes the specific Playwright operation for a single step.
        """
        logger.info(f"Executing step {step.id} ({step.action_type})...")

        if step.action_type == "navigate":
            await self._execute_navigate(page, step)
        elif step.action_type == "fill":
            await self._execute_fill(page, step)
        elif step.action_type == "upload":
            await self._execute_upload(page, step)
        elif step.action_type == "click":
            await self._execute_click(page, step)
        elif step.action_type == "wait_user":
            logger.info(f"User action required: {step.description}")

    async def _execute_navigate(self, page: Page, step: ActionStep) -> None:
        if not step.value:
            raise ValueError("Navigation step missing target URL value.")

        # Check LinkedIn domain validation safeguard
        if "linkedin" in step.value and not is_linkedin_url(step.value):
            raise SafetyGuardViolationError(
                f"Blocked navigation to untrusted LinkedIn domain: {step.value}"
            )

        await page.goto(step.value, wait_until="domcontentloaded")

    async def _execute_fill(self, page: Page, step: ActionStep) -> None:
        if not step.target_selector or step.value is None:
            raise ValueError("Fill step missing selector or fill value.")

        # Visually highlight target element
        await highlight_element(page, step.target_selector)

        # Mimic human typing delays
        await page.focus(step.target_selector)
        await page.fill(step.target_selector, "")

        val_str = str(step.value)
        for char in val_str:
            await page.keyboard.type(char)
            await asyncio.sleep(self.config.typing_speed_ms / 1000.0)

    async def _execute_upload(self, page: Page, step: ActionStep) -> None:
        if not step.target_selector or not step.value:
            raise ValueError("Upload step missing selector or local file path.")

        # Visually highlight file element
        await highlight_element(page, step.target_selector)
        await page.set_input_files(step.target_selector, step.value)

    async def _execute_click(self, page: Page, step: ActionStep) -> None:
        if not step.target_selector:
            raise ValueError("Click step missing target selector.")

        await highlight_element(page, step.target_selector)
        await page.click(step.target_selector)

    def _check_message_limits(self) -> None:
        """
        Safeguard check preventing mass direct messaging.
        """
        self._message_count += 1
        if self._message_count > self.config.max_messaging_limit:
            raise SafetyGuardViolationError(
                f"Action blocked: Exceeded the maximum of {self.config.max_messaging_limit} messaging operations "
                "per session to prevent spam behavior."
            )
