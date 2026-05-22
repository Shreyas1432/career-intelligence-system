from .types import CanonicalSkill, SkillCategory

DEFAULT_SKILL_TAXONOMY: tuple[CanonicalSkill, ...] = (
    CanonicalSkill(
        name="Python",
        category=SkillCategory.PROGRAMMING,
        aliases=("py", "python3"),
    ),
    CanonicalSkill(
        name="SQL",
        category=SkillCategory.DATA_AI,
        aliases=("structured query language", "sql querying", "sql queries"),
    ),
    CanonicalSkill(
        name="Spark",
        category=SkillCategory.DATA_AI,
        aliases=("apache spark", "pyspark", "spark sql", "spark streaming", "databricks spark"),
    ),
    CanonicalSkill(
        name="Databricks",
        category=SkillCategory.DATA_AI,
        aliases=("azure databricks", "databricks lakehouse"),
    ),
    CanonicalSkill(
        name="Machine Learning",
        category=SkillCategory.DATA_AI,
        aliases=("ml", "predictive modeling", "predictive modelling"),
    ),
    CanonicalSkill(
        name="Generative AI",
        category=SkillCategory.DATA_AI,
        aliases=("genai", "gen ai", "generative artificial intelligence"),
    ),
    CanonicalSkill(
        name="Large Language Models",
        category=SkillCategory.DATA_AI,
        aliases=("llm", "llms", "large language model"),
    ),
    CanonicalSkill(
        name="Natural Language Processing",
        category=SkillCategory.DATA_AI,
        aliases=("nlp", "text analytics", "text mining"),
    ),
    CanonicalSkill(
        name="MLOps",
        category=SkillCategory.DATA_AI,
        aliases=("ml ops", "model operations", "model deployment"),
    ),
    CanonicalSkill(
        name="Data Engineering",
        category=SkillCategory.DATA_AI,
        aliases=("data pipelines", "etl", "elt", "pipeline engineering"),
    ),
    CanonicalSkill(
        name="Data Visualization",
        category=SkillCategory.ANALYTICS,
        aliases=("data visualisation", "dashboarding", "dashboards"),
    ),
    CanonicalSkill(
        name="Power BI",
        category=SkillCategory.ANALYTICS,
        aliases=("powerbi", "microsoft power bi"),
    ),
    CanonicalSkill(
        name="Tableau",
        category=SkillCategory.ANALYTICS,
        aliases=("tableau desktop", "tableau server"),
    ),
    CanonicalSkill(
        name="ERP",
        category=SkillCategory.ENTERPRISE_SYSTEMS,
        aliases=(
            "enterprise resource planning",
            "oracle fusion",
            "oracle fusion cloud",
            "oracle erp cloud",
            "sap erp",
            "sap s4 hana",
            "sap s/4hana",
            "workday financials",
            "microsoft dynamics 365",
        ),
    ),
    CanonicalSkill(
        name="CRM",
        category=SkillCategory.ENTERPRISE_SYSTEMS,
        aliases=(
            "customer relationship management",
            "salesforce",
            "salesforce crm",
            "dynamics crm",
        ),
    ),
    CanonicalSkill(
        name="SAP",
        category=SkillCategory.ENTERPRISE_SYSTEMS,
        aliases=("sap ecc", "sap hana"),
    ),
    CanonicalSkill(
        name="Supply Chain Analytics",
        category=SkillCategory.SUPPLY_CHAIN,
        aliases=(
            "procurement analytics",
            "supply chain analysis",
            "supply chain reporting",
            "spend analytics",
            "supplier analytics",
            "category analytics",
        ),
    ),
    CanonicalSkill(
        name="Procurement",
        category=SkillCategory.PROCUREMENT,
        aliases=("strategic sourcing", "sourcing", "purchasing", "vendor management"),
    ),
    CanonicalSkill(
        name="Category Management",
        category=SkillCategory.PROCUREMENT,
        aliases=("category strategy", "category planning"),
    ),
    CanonicalSkill(
        name="Supplier Management",
        category=SkillCategory.PROCUREMENT,
        aliases=("supplier relationship management", "srm", "supplier performance"),
    ),
    CanonicalSkill(
        name="Demand Planning",
        category=SkillCategory.SUPPLY_CHAIN,
        aliases=("demand forecasting", "forecasting", "inventory forecasting"),
    ),
    CanonicalSkill(
        name="Inventory Management",
        category=SkillCategory.SUPPLY_CHAIN,
        aliases=("stock management", "inventory optimization", "inventory optimisation"),
    ),
    CanonicalSkill(
        name="AWS",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("amazon web services", "aws cloud"),
    ),
    CanonicalSkill(
        name="Azure",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("microsoft azure", "azure cloud"),
    ),
    CanonicalSkill(
        name="Google Cloud",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("gcp", "google cloud platform"),
    ),
    CanonicalSkill(
        name="Kubernetes",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("k8s", "kube"),
    ),
    CanonicalSkill(
        name="Docker",
        category=SkillCategory.CLOUD_INFRASTRUCTURE,
        aliases=("containerization", "containerisation", "containers"),
    ),
    CanonicalSkill(
        name="Cybersecurity",
        category=SkillCategory.SECURITY,
        aliases=("cyber security", "information security", "infosec"),
    ),
    CanonicalSkill(
        name="Product Management",
        category=SkillCategory.PRODUCT,
        aliases=("product strategy", "roadmapping", "product roadmap"),
    ),
    CanonicalSkill(
        name="Stakeholder Management",
        category=SkillCategory.BUSINESS,
        aliases=("stakeholder engagement", "executive communication"),
    ),
)
