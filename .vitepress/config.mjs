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
            { text: "OOD_intro", link: "/discussions/OOD_intro/" },
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
            { text: "野生数据集中的OOD检测", link: "/discussions/OOD-in-wild-datasets/"}
          ],
        },
        {
          text: "第四次",
          collapsed: false,
          items: [
            {
              text: "Influence_function(影响函数)",
              link: "/discussions/Influence_function/",
            },
          ],
        },
        {
          text: "第五次",
          collapsed: false,
          items: [
            {
              text: "CLIP 简介",
              link: "/discussions/CLIP_intro/",
            },
            {
              text: "DBA&CBA_in_graph",
              link: "/discussions/DBA&CBA _in_graph/",
            },
          ],
        },
      ],
    },
    search: {
      provider: "local",
    },
  },
});
