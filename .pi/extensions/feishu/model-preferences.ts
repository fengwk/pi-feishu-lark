export const THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"] as const;

export type ThinkingLevel = (typeof THINKING_LEVELS)[number];

type ModelLike = {
  provider: string;
  id: string;
  name?: string;
  reasoning?: boolean;
  thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>;
};

export type ScopedModelChoice<TModel extends ModelLike = ModelLike> = {
  model: TModel;
  thinkingLevel?: ThinkingLevel;
};

type ParsedModelResult<TModel extends ModelLike> = {
  model: TModel | undefined;
  thinkingLevel?: ThinkingLevel;
};

export function isThinkingLevel(value: string | undefined): value is ThinkingLevel {
  return Boolean(value && THINKING_LEVELS.includes(value as ThinkingLevel));
}

export function getSupportedThinkingLevels<TModel extends ModelLike>(model: TModel | undefined): ThinkingLevel[] {
  if (!model?.reasoning) return ["off"];

  return THINKING_LEVELS.filter((level) => {
    const mapped = model.thinkingLevelMap?.[level];
    if (mapped === null) return false;
    if (level === "xhigh") return mapped !== undefined;
    return true;
  });
}

export function clampThinkingLevel<TModel extends ModelLike>(model: TModel | undefined, level: ThinkingLevel): ThinkingLevel {
  const availableLevels = getSupportedThinkingLevels(model);
  if (availableLevels.includes(level)) return level;

  const requestedIndex = THINKING_LEVELS.indexOf(level);
  if (requestedIndex === -1) return availableLevels[0] || "off";

  for (let i = requestedIndex; i < THINKING_LEVELS.length; i += 1) {
    const candidate = THINKING_LEVELS[i];
    if (availableLevels.includes(candidate)) return candidate;
  }

  for (let i = requestedIndex - 1; i >= 0; i -= 1) {
    const candidate = THINKING_LEVELS[i];
    if (availableLevels.includes(candidate)) return candidate;
  }

  return availableLevels[0] || "off";
}

export function resolveEnabledModelScope<TModel extends ModelLike>(
  patterns: string[] | undefined,
  availableModels: TModel[],
): Array<ScopedModelChoice<TModel>> {
  if (!patterns?.length) {
    return availableModels.map((model) => ({ model }));
  }

  const scopedModels: Array<ScopedModelChoice<TModel>> = [];
  for (const pattern of patterns) {
    if (hasGlob(pattern)) {
      const { basePattern, thinkingLevel } = splitPatternThinkingLevel(pattern);
      const matches = availableModels.filter((model) => matchesGlobPattern(basePattern, model));
      for (const model of matches) {
        if (!hasScopedModel(scopedModels, model)) scopedModels.push({ model, thinkingLevel });
      }
      continue;
    }

    const parsed = parseModelPattern(pattern, availableModels);
    if (parsed.model && !hasScopedModel(scopedModels, parsed.model)) {
      scopedModels.push(parsed);
    }
  }

  return scopedModels;
}

function hasGlob(pattern: string) {
  return pattern.includes("*") || pattern.includes("?") || pattern.includes("[");
}

function splitPatternThinkingLevel(pattern: string): { basePattern: string; thinkingLevel?: ThinkingLevel } {
  const colonIndex = pattern.lastIndexOf(":");
  if (colonIndex === -1) return { basePattern: pattern };

  const suffix = pattern.substring(colonIndex + 1);
  if (!isThinkingLevel(suffix)) return { basePattern: pattern };

  return {
    basePattern: pattern.substring(0, colonIndex),
    thinkingLevel: suffix,
  };
}

function parseModelPattern<TModel extends ModelLike>(pattern: string, availableModels: TModel[]): ParsedModelResult<TModel> {
  const exactMatch = tryMatchModel(pattern, availableModels);
  if (exactMatch) {
    return { model: exactMatch };
  }

  const colonIndex = pattern.lastIndexOf(":");
  if (colonIndex === -1) {
    return { model: undefined };
  }

  const prefix = pattern.substring(0, colonIndex);
  const suffix = pattern.substring(colonIndex + 1);
  if (isThinkingLevel(suffix)) {
    const result = parseModelPattern(prefix, availableModels);
    if (result.model) {
      return {
        model: result.model,
        thinkingLevel: suffix,
      };
    }
    return result;
  }

  return parseModelPattern(prefix, availableModels);
}

function tryMatchModel<TModel extends ModelLike>(modelPattern: string, availableModels: TModel[]): TModel | undefined {
  const exactMatch = findExactModelReferenceMatch(modelPattern, availableModels);
  if (exactMatch) return exactMatch;

  const normalizedPattern = modelPattern.toLowerCase();
  const matches = availableModels.filter((model) =>
    model.id.toLowerCase().includes(normalizedPattern)
    || model.name?.toLowerCase().includes(normalizedPattern),
  );
  if (!matches.length) return undefined;

  const aliases = matches.filter((model) => isAlias(model.id));
  if (aliases.length) {
    aliases.sort((a, b) => b.id.localeCompare(a.id));
    return aliases[0];
  }

  matches.sort((a, b) => b.id.localeCompare(a.id));
  return matches[0];
}

function findExactModelReferenceMatch<TModel extends ModelLike>(modelReference: string, availableModels: TModel[]): TModel | undefined {
  const trimmedReference = modelReference.trim();
  if (!trimmedReference) return undefined;

  const normalizedReference = trimmedReference.toLowerCase();
  const canonicalMatches = availableModels.filter(
    (model) => `${model.provider}/${model.id}`.toLowerCase() === normalizedReference,
  );
  if (canonicalMatches.length === 1) return canonicalMatches[0];
  if (canonicalMatches.length > 1) return undefined;

  const slashIndex = trimmedReference.indexOf("/");
  if (slashIndex !== -1) {
    const provider = trimmedReference.substring(0, slashIndex).trim();
    const modelId = trimmedReference.substring(slashIndex + 1).trim();
    if (provider && modelId) {
      const providerMatches = availableModels.filter(
        (model) => model.provider.toLowerCase() === provider.toLowerCase() && model.id.toLowerCase() === modelId.toLowerCase(),
      );
      if (providerMatches.length === 1) return providerMatches[0];
      if (providerMatches.length > 1) return undefined;
    }
  }

  const idMatches = availableModels.filter((model) => model.id.toLowerCase() === normalizedReference);
  return idMatches.length === 1 ? idMatches[0] : undefined;
}

function isAlias(id: string) {
  if (id.endsWith("-latest")) return true;
  return !/-\d{8}$/.test(id);
}

function hasScopedModel<TModel extends ModelLike>(scopedModels: Array<ScopedModelChoice<TModel>>, model: TModel) {
  return scopedModels.some((item) => item.model.provider === model.provider && item.model.id === model.id);
}

function matchesGlobPattern<TModel extends ModelLike>(pattern: string, model: TModel) {
  const fullId = `${model.provider}/${model.id}`;
  return matchGlob(pattern, fullId) || matchGlob(pattern, model.id);
}

function matchGlob(pattern: string, value: string) {
  return globToRegExp(pattern).test(value);
}

function globToRegExp(pattern: string) {
  let regex = "^";
  for (let i = 0; i < pattern.length; i += 1) {
    const char = pattern[i];
    if (char === "*") {
      regex += ".*";
      continue;
    }
    if (char === "?") {
      regex += ".";
      continue;
    }
    if (char === "[") {
      const end = pattern.indexOf("]", i + 1);
      if (end === -1) {
        regex += "\\[";
        continue;
      }
      let content = pattern.slice(i + 1, end);
      if (content.startsWith("!")) {
        content = `^${content.slice(1)}`;
      }
      regex += `[${content}]`;
      i = end;
      continue;
    }
    regex += escapeRegExpChar(char);
  }
  regex += "$";
  return new RegExp(regex, "i");
}

function escapeRegExpChar(char: string) {
  return /[\\^$+?.()|{}]/.test(char) ? `\\${char}` : char;
}
