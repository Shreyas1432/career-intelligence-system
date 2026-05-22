# tests/utils/scraping_mocks.py

MOCK_GREENHOUSE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Acme Corp - Senior Python Engineer (Greenhouse)</title>
    <meta name="description" content="Greenhouse Job Board Posting">
</head>
<body>
    <div id="wrapper">
        <header class="gh-header">
            <h1>Senior Python Engineer</h1>
            <span class="location">Remote, US</span>
        </header>
        <main>
            <div id="content" class="greenhouse-embed" data-board="acme">
                <h2>About the Role</h2>
                <p>We are looking for a Senior Python Engineer to scale our data pipelines. You will use Python and PySpark.</p>

                <h2>Requirements</h2>
                <ul>
                    <li>5+ years of experience with Python</li>
                    <li>Experience with Apache Spark or PySpark</li>
                    <li>Strong SQL skills</li>
                </ul>

                <h2>Benefits</h2>
                <p>Visa sponsorship is available for qualified candidates.</p>
                <div class="apply-button">
                    <a href="https://boards.greenhouse.io/acme/jobs/12345/apply?gh_jid=12345">Apply Now</a>
                </div>
            </div>
        </main>
    </div>
</body>
</html>
"""

MOCK_LEVER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Lever Inc - Backend Developer</title>
</head>
<body>
    <div class="lever-posting" data-id="54321">
        <div class="posting-header">
            <h2>Backend Developer</h2>
            <div class="categories">
                <span class="department">Engineering</span>
                <span class="workplaceTypes">Hybrid</span>
                <span class="location">San Francisco, CA</span>
                <span class="commitment">Full-time</span>
            </div>
        </div>
        <div class="section-wrapper lever-job">
            <h3>Description</h3>
            <p>Develop robust APIs using Python and FastAPI. Keep systems optimized.</p>
            <h3>Requirements</h3>
            <ul>
                <li>3+ years backend software development experience</li>
                <li>Familiarity with FastAPI and SQLAlchemy</li>
            </ul>
        </div>
        <div class="posting-apply">
            <a href="https://jobs.lever.co/leverinc/54321/apply" class="postings-btn template-btn-submit">Submit Application</a>
        </div>
        <div class="footer">
            <p>Jobs hosted by lever</p>
        </div>
    </div>
</body>
</html>
"""

MOCK_MALFORMED_HTML = """
<html><head><title>Broken Job Posting</title></head><body>
<div><h1>Software Developer</h1>
<p>Missing close tags everywhere.
<script>document.write("Noisy script context");
<div class="content">No closing tags for outer div, no closing body, or html.
"""

MOCK_GREENHOUSE_EXTRACTION = {
    "company": "Acme Corp",
    "title": "Senior Python Engineer",
    "skills": ["Python", "Spark", "SQL"],
    "experience_required": "5+ years",
    "location": "Remote, US",
    "visa_signal": "sponsorship_available",
    "employment_type": "full_time",
    "domain": "software_engineering",
    "confidence_score": 0.95,
}

MOCK_LEVER_EXTRACTION = {
    "company": "Lever Inc",
    "title": "Backend Developer",
    "skills": ["Python", "FastAPI", "SQLAlchemy"],
    "experience_required": "3+ years",
    "location": "San Francisco, CA",
    "visa_signal": "unknown",
    "employment_type": "full_time",
    "domain": "software_engineering",
    "confidence_score": 0.90,
}
