import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "zh-CN",
  title: "Deep Learning Discussion",
  lastUpdated: true,
  cleanUrls: true,
  base: "/dl-discussion/",
  markdown: {
    math: true,
  },
  themeConfig: {
    siteTitle: "Deep Learning Discussion",
    nav: [
      { text: "研讨会记录", link: "/discussions/" },
      { text: "成员", link: "/members" },
    ],
    sidebar: {
      discussions: [
        {
          text: "第一次",
          collapsed: false,
          items: [
            { text: "GNN 简介", link: "/discussions/GNN-intro/" },
            { text: "some math for GNN", link: "/discussions/GNN_math/" },
            { text: "AC 简介", link: "/discussions/AC-intro/" },
          ],
        },
        {
          text: "第二次",
          collapsed: false,
          items: [
            { text: "LLM 初探", link: "/discussions/llm-quick-look/" },
            { text: "FER1", link: "/discussions/FER1/" },
          ],
        },
        {
          text: "第三次",
          collapsed: false,
          items: [
            { text: "LAD-GNN(注意力与蒸馏)", link: "/discussions/LAD-GNN/" },
          ],
        },
      ],
    },
    search: {
      provider: "local",
    },
  },
});
