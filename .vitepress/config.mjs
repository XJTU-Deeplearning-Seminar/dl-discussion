import { defineConfig } from "vitepress";

export default defineConfig({
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "研讨会记录", link: "/discussions/" },
    ],
    sidebar: {
      discussions: [
        {
          text: "第一次",
          collapsed: false,
          items: [
            { text: "LLM 初探", link: "/discussions/llm-quick-lok/" },
            { text: "LLM 详解", link: "/discussions/llm-in-depth/" },
          ],
        },
        {
          text: "第二次",
          collapsed: false,
          items: [{ text: "LLM 初探", link: "/discussions/llm-quick-look/" }],
        },
      ],
    },
  },
});
