# 计试机器学习研讨会记录

## 上传指南

### 关于 Git

建议使用 rebase 的工作流。 对于修改自己 markdown 文件的 commit，Commit message 推荐为 `docs: add/modify 文章标题`，其他类型的提交请参考 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/)。

简单教程：每次自己 Commit（提交）之后，在终端运行 `git pull --rebase`（或者将 VSCode 设置中的 "Git: Rebase When Sync" 打开后运行 Sync （同步）），然后解决冲突（如果有的话），再运行 `git push`。

仓库目前打开了 "Require linear history" 选项，同时禁止任何人 `force push`。因此请不要使用 `merge`，而是使用 `rebase`。

### 添加 markdown 文件

1. 在 discussions 文件夹下新建一个文件夹（例如 `GNN-intro`），这个文件夹名会成为访问网址的一部分，不建议有空格。
2. 都 `GNN-intro` 文件夹下新建一个 `index.md` 文件，其中放入你的 markdown 内容。
3. 如果有图片，放在 `GNN-intro` 文件夹下，引用时使用相对路径。

### 更改配置

打开 `.vitepress/config.mjs` 文件，可以看到若干个形似下面示例的 `sidebar.discussions` 配置。

```javascript
{
  text: "第一次", // 侧边栏小标题（每次研讨会一个小标题）
  collapsed: false,
  items: [
    { text: "GNN 简介", link: "/discussions/GNN-intro/" },
    { text: "ABC 示例", link: "/discussions/ABC-example/" }, // 侧边栏条目
  ],
},
```

如果是对某次已存在的研讨会进行添加，只需要在 `items` 数组中添加一个新的对象即可。如果是新的研讨会，需要新建一个如上对象的数组，然后添加到 `discussions` 数组中。

`items` 数组中的 `text` 是侧边栏条目的名字，`link` 是对应的 markdown 文件的路径，记得前后的反斜杠。

然后提交推送即可。关于 Git，请看 [关于 Git](#关于-git)。

推送后有 Actions 对 `config.mjs` 文件格式化。
