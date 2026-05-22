import re
from collections.abc import Iterable, Mapping, Sequence

from .taxonomy import DEFAULT_SKILL_TAXONOMY
from .types import CanonicalSkill, NormalizedSkill, SkillCategory

TOKEN_SEPARATOR_PATTERN = re.compile(r"[\s_\-/]+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s+#.]")


def normalize_lookup_key(value: str) -> str:
    """
    Convert skill text into a stable lowercase lookup key.
    """
    stripped = value.strip().casefold()
    without_punctuation = PUNCTUATION_PATTERN.sub(" ", stripped)
    normalized_separators = TOKEN_SEPARATOR_PATTERN.sub(" ", without_punctuation)
    return " ".join(normalized_separators.split())


def normalize_display_name(value: str) -> str:
    """
    Normalize unknown skills for display without inventing a taxonomy match.
    """
    return " ".join(value.strip().split())


class SkillNormalizer:
    """
    Deterministic skill synonym normalizer backed by an in-memory taxonomy.
    """

    def __init__(
        self,
        taxonomy: Sequence[CanonicalSkill] = DEFAULT_SKILL_TAXONOMY,
        extra_aliases: Mapping[str, str] | None = None,
    ):
        self._taxonomy = tuple(taxonomy)
        self._canonical_by_name = {entry.name: entry for entry in self._taxonomy}
        self._lookup: dict[str, CanonicalSkill] = {}
        self._load_taxonomy(self._taxonomy)
        self._load_extra_aliases(extra_aliases or {})

    @property
    def taxonomy(self) -> tuple[CanonicalSkill, ...]:
        return self._taxonomy

    def normalize(self, skill: str) -> NormalizedSkill | None:
        """
        Normalize one skill string to a canonical skill result.
        """
        if not isinstance(skill, str):
            raise TypeError("skill must be a string")

        original = normalize_display_name(skill)
        if not original:
            return None

        key = normalize_lookup_key(original)
        canonical = self._lookup.get(key)
        if canonical is None:
            return NormalizedSkill(
                original=original,
                canonical=original,
                category=SkillCategory.OTHER,
                matched_alias=None,
            )

        matched_alias = original if canonical.name != original else None
        return NormalizedSkill(
            original=original,
            canonical=canonical.name,
            category=canonical.category,
            matched_alias=matched_alias,
        )

    def normalize_many(self, skills: Iterable[str]) -> list[NormalizedSkill]:
        """
        Normalize a skill collection and dedupe by canonical skill key.
        """
        normalized: list[NormalizedSkill] = []
        seen: set[str] = set()

        for skill in skills:
            result = self.normalize(skill)
            if result is None:
                continue

            key = normalize_lookup_key(result.canonical)
            if key in seen:
                continue

            seen.add(key)
            normalized.append(result)

        return normalized

    def canonicalize_many(self, skills: Iterable[str]) -> list[str]:
        """
        Return only canonical skill names for downstream matching.
        """
        return [skill.canonical for skill in self.normalize_many(skills)]

    def _load_taxonomy(self, taxonomy: Sequence[CanonicalSkill]) -> None:
        for entry in taxonomy:
            self._register(entry.name, entry)
            for alias in entry.aliases:
                self._register(alias, entry)

    def _load_extra_aliases(self, extra_aliases: Mapping[str, str]) -> None:
        for alias, canonical_name in extra_aliases.items():
            canonical = self._canonical_by_name.get(canonical_name)
            if canonical is None:
                canonical = CanonicalSkill(
                    name=normalize_display_name(canonical_name),
                    category=SkillCategory.OTHER,
                    aliases=(alias,),
                )
                self._canonical_by_name[canonical.name] = canonical

            self._register(alias, canonical)

    def _register(self, alias: str, canonical: CanonicalSkill) -> None:
        key = normalize_lookup_key(alias)
        if key:
            self._lookup[key] = canonical


default_skill_normalizer = SkillNormalizer()


def normalize_skill(skill: str) -> NormalizedSkill | None:
    return default_skill_normalizer.normalize(skill)


def normalize_skills(skills: Iterable[str]) -> list[NormalizedSkill]:
    return default_skill_normalizer.normalize_many(skills)


def canonicalize_skills(skills: Iterable[str]) -> list[str]:
    return default_skill_normalizer.canonicalize_many(skills)
