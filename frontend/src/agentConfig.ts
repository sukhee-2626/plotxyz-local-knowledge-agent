export type FieldType = "text" | "textarea" | "number" | "checkbox";

export interface AgentField {
  name: string;
  label: string;
  type: FieldType;
  rows?: number;
  placeholder?: string;
}

export interface AgentConfig {
  slug: string;
  title: string;
  tagline: string;
  purpose: string;
  accent: string;
  features: string[];
  limitations: string[];
  fields: AgentField[];
  sample: Record<string, string | number | boolean>;
  repo: string;
}

export const agent: AgentConfig = {
  "slug": "plotxyz-local-knowledge-agent",
  "title": "Local Knowledge",
  "tagline": "Ask your own documents, with receipts.",
  "purpose": "Indexes user-approved documents in memory, retrieves relevant passages by lexical relevance, and returns extractive answers with source titles.",
  "accent": "#34d399",
  "features": [
    "Passage-level keyword retrieval",
    "Source-cited extractive answers",
    "Relevance scoring"
  ],
  "limitations": [
    "Keyword retrieval only; no embeddings, vector database, or external search.",
    "Documents are not persisted between requests."
  ],
  "fields": [
    {
      "name": "documents",
      "label": "Documents (separate with ###)",
      "type": "textarea",
      "rows": 8,
      "placeholder": ""
    },
    {
      "name": "query",
      "label": "Question",
      "type": "textarea",
      "rows": 3,
      "placeholder": ""
    },
    {
      "name": "top_k",
      "label": "Passages to return",
      "type": "number",
      "rows": 4,
      "placeholder": ""
    }
  ],
  "sample": {
    "documents": "### Release notes\nVersion 2 adds offline mode and a faster index. The offline mode stores documents locally.\n\n### Security\nAll local documents stay on the device. Export requires confirmation.",
    "query": "offline documents local",
    "top_k": 2
  },
  "repo": "https://github.com/sukhee-2626/plotxyz-local-knowledge-agent"
};
