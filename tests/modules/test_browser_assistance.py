from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.modules.automation import (
    ActionStep,
    BrowserAssistanceConfig,
    BrowserAssistanceEngine,
    FormMapping,
    SafetyGuardViolationError,
    WorkflowStepFailedError,
)
from src.modules.automation.browser import (
    classify_field,
    generate_css_selector,
    is_linkedin_url,
)


def test_config_and_defaults() -> None:
    config = BrowserAssistanceConfig()
    assert config.headless is False
    assert config.typing_speed_ms == 50
    assert config.step_delay_seconds == 0.5
    assert config.max_messaging_limit == 3


def test_is_linkedin_url_checker() -> None:
    assert is_linkedin_url("https://www.linkedin.com/in/john-doe") is True
    assert is_linkedin_url("https://linkedin.com/jobs") is True
    assert is_linkedin_url("https://example.com") is False
    assert is_linkedin_url("") is False


def test_field_classification() -> None:
    # Test email
    email_field = {
        "id": "user-email-input",
        "name": "email",
        "placeholder": "Enter your email",
        "labelText": "Email Address",
        "type": "text",
        "tagName": "input",
    }
    assert classify_field(email_field) == "email"

    # Test phone
    phone_field = {
        "id": "phone-num",
        "name": "tel",
        "placeholder": "Phone",
        "labelText": "Mobile Number",
        "type": "tel",
        "tagName": "input",
    }
    assert classify_field(phone_field) == "phone"

    # Test resume
    resume_field = {
        "id": "resume-upload",
        "name": "cv_file",
        "placeholder": "",
        "labelText": "Upload CV",
        "type": "file",
        "tagName": "input",
    }
    assert classify_field(resume_field) == "resume_file"

    # Test cover letter
    cl_field = {
        "id": "notes",
        "name": "msg_hiring_manager",
        "placeholder": "",
        "labelText": "Cover Letter Notes",
        "type": "text",
        "tagName": "textarea",
    }
    assert classify_field(cl_field) == "cover_letter"

    # Test fallback / unmapped
    unknown_field = {
        "id": "unknown-attr",
        "name": "custom_field",
        "placeholder": "some val",
        "labelText": "Favorite Color",
        "type": "text",
        "tagName": "input",
    }
    assert classify_field(unknown_field) is None


def test_css_selector_generator() -> None:
    assert generate_css_selector({"id": "submit-btn", "tagName": "button"}) == "#submit-btn"
    assert (
        generate_css_selector({"name": "email_input", "tagName": "input"})
        == "input[name='email_input']"
    )
    assert (
        generate_css_selector({"type": "text", "placeholder": "Search...", "tagName": "input"})
        == "input[type='text'][placeholder='Search...']"
    )
    assert generate_css_selector({"index": 2, "tagName": "input"}) == "input:nth-of-type(3)"


def test_create_linkedin_workflow() -> None:
    engine = BrowserAssistanceEngine()
    workflow = engine.create_linkedin_workflow("johndoe", "experience")

    assert workflow.metadata["type"] == "linkedin_navigation"
    assert workflow.metadata["username"] == "johndoe"
    assert workflow.metadata["section"] == "experience"
    assert len(workflow.steps) == 2
    assert workflow.steps[0].action_type == "wait_user"
    assert workflow.steps[1].action_type == "navigate"
    val = workflow.steps[1].value
    assert isinstance(val, str)
    assert "linkedin.com/in/johndoe/details/experience/" in val


@pytest.mark.asyncio
async def test_engine_lifecycle_close() -> None:
    engine = BrowserAssistanceEngine()

    with patch("src.modules.automation.browser.BrowserManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.start = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager_class.return_value = mock_manager

        await engine.start()
        assert engine._is_running is True

        await engine.close()
        mock_manager.close.assert_called_once()
        assert engine._is_running is False


@pytest.mark.asyncio
async def test_execute_workflow_approved_steps() -> None:
    engine = BrowserAssistanceEngine(config=BrowserAssistanceConfig(step_delay_seconds=0.0))

    # Mock page elements
    mock_page = MagicMock()
    mock_page.goto = AsyncMock()
    mock_page.focus = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.keyboard = MagicMock()
    mock_page.keyboard.type = AsyncMock()
    mock_page.set_input_files = AsyncMock()
    mock_page.evaluate = AsyncMock()

    # Mock locator
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=1)
    mock_page.locator.return_value = mock_locator

    workflow = engine.create_linkedin_workflow("johndoe", "skills")

    steps_yielded = []
    # Execute the workflow runner generator
    async for step in engine.run_workflow(workflow, mock_page):
        # Approve all steps
        step.approve()
        steps_yielded.append(step)

    assert len(steps_yielded) == 2
    assert all(step.status == "completed" for step in steps_yielded)
    mock_page.goto.assert_called_once_with(
        "https://www.linkedin.com/in/johndoe/details/skills/", wait_until="domcontentloaded"
    )


@pytest.mark.asyncio
async def test_execute_workflow_skipped_rejected_steps() -> None:
    engine = BrowserAssistanceEngine(config=BrowserAssistanceConfig(step_delay_seconds=0.0))

    mock_page = MagicMock()
    mock_page.goto = AsyncMock()
    workflow = engine.create_linkedin_workflow("johndoe", "skills")

    steps_yielded = []
    async for step in engine.run_workflow(workflow, mock_page):
        # Reject step 1, approve step 2
        if step.id == "check-login":
            step.reject()
        else:
            step.approve()
        steps_yielded.append(step)

    assert steps_yielded[0].status == "rejected"
    assert steps_yielded[1].status == "completed"


@pytest.mark.asyncio
async def test_untrusted_linkedin_url_fails() -> None:
    engine = BrowserAssistanceEngine()

    mock_page = MagicMock()
    step = ActionStep(
        id="unsafe-nav",
        action_type="navigate",
        description="Dangerous URL",
        value="https://www.linkedin-scam.com/in/johndoe",
        status="approved",
    )

    with pytest.raises(
        SafetyGuardViolationError, match="Blocked navigation to untrusted LinkedIn domain"
    ):
        await engine._execute_single_step(mock_page, step)


@pytest.mark.asyncio
async def test_workflow_step_failure_handling() -> None:
    engine = BrowserAssistanceEngine(config=BrowserAssistanceConfig(step_delay_seconds=0.0))

    mock_page = MagicMock()
    # Mock goto to raise error
    mock_page.goto = AsyncMock(side_effect=RuntimeError("Connection Timeout"))

    workflow = engine.create_linkedin_workflow("johndoe", "skills")
    # Mark first step as completed so we skip directly to navigation
    workflow.steps[0].status = "completed"

    async def consume() -> None:
        async for step in engine.run_workflow(workflow, mock_page):
            step.approve()

    with pytest.raises(
        WorkflowStepFailedError, match="Workflow execution failed at step navigate-section"
    ):
        await consume()


@pytest.mark.asyncio
async def test_job_application_form_scanning() -> None:
    engine = BrowserAssistanceEngine()
    form_data = FormMapping(
        email="test@example.com",
        first_name="Jane",
        last_name="Doe",
        phone="555-1234",
        linkedin_url="https://www.linkedin.com/in/janedoe",
        cover_letter="Interested in the backend lead position.",
    )

    with patch("src.modules.automation.browser.BrowserManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.start = AsyncMock()

        # Mock Page context manager
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()

        # Mock detected fields
        detected_fields_data = [
            {
                "id": "first-name-input",
                "name": "first_name",
                "type": "text",
                "placeholder": "First Name",
                "labelText": "First Name",
                "tagName": "input",
            },
            {
                "id": "email-input",
                "name": "email",
                "type": "email",
                "placeholder": "Email",
                "labelText": "Email Address",
                "tagName": "input",
            },
            {
                "id": "cv-file",
                "name": "resume",
                "type": "file",
                "placeholder": "",
                "labelText": "Upload Resume",
                "tagName": "input",
            },
        ]

        # In mock context, page is returned by async context manager
        # Mock the context manager __aenter__ to return mock_page
        mock_page_context = MagicMock()
        mock_page_context.__aenter__ = AsyncMock(return_value=mock_page)
        mock_page_context.__aexit__ = AsyncMock(return_value=None)

        mock_manager.page.return_value = mock_page_context
        mock_manager_class.return_value = mock_manager

        await engine.start()

        with patch(
            "src.modules.automation.browser.detect_fields",
            AsyncMock(return_value=detected_fields_data),
        ):
            workflow = await engine.create_job_application_workflow(
                url="https://careers.example.com/apply",
                form_data=form_data,
                resume_path="/path/to/resume.pdf",
            )

            assert (
                len(workflow.steps) == 5
            )  # Navigation + 2 filled inputs + 1 resume upload + review/submit
            assert workflow.steps[0].action_type == "navigate"
            assert workflow.steps[1].action_type == "fill"
            assert "Jane" in workflow.steps[1].description
            assert workflow.steps[2].action_type == "fill"
            assert "test@example.com" in workflow.steps[2].description
            assert workflow.steps[3].action_type == "upload"
            assert "/path/to/resume.pdf" in workflow.steps[3].description
            assert workflow.steps[4].action_type == "wait_user"


def test_anti_spam_safeguard() -> None:
    engine = BrowserAssistanceEngine(config=BrowserAssistanceConfig(max_messaging_limit=2))

    # Trigger message counter check
    engine._check_message_limits()
    engine._check_message_limits()

    # Third time should raise safety guard violation
    with pytest.raises(
        SafetyGuardViolationError, match="Exceeded the maximum of 2 messaging operations"
    ):
        engine._check_message_limits()
