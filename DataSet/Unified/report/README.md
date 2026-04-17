# report_4_16

LaTeX 实验报告主文件：

- `report_4_16.tex`

当前报告使用 `ctex`，本机已通过用户目录安装完整 TinyTeX。

推荐编译命令：

```bash
/home/lidz/.TinyTeX-full/bin/x86_64-linux/xelatex report_4_16.tex
/home/lidz/.TinyTeX-full/bin/x86_64-linux/xelatex report_4_16.tex
```

如果已经把 TinyTeX 的 `bin` 目录加入了 `PATH`，也可以直接使用：

```bash
xelatex report_4_16.tex
```

当前本机实际使用的 TeX 发行版目录：

- `/home/lidz/.TinyTeX-full`

采用该方案的原因：

- 系统自带 `lualatex` 缺少中文宏包，无法正确处理 `ctex`
- 通过 GitHub 下载的完整 TinyTeX-2 daily 包已包含 `ctex` 与 `xelatex`
- 该方案不需要管理员权限
