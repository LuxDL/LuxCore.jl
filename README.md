# LuxCore

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://lux.csail.mit.edu/dev/api/Building_Blocks/LuxCore)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxCore)

[![Build status](https://badge.buildkite.com/702f7908a08898971896c9bf5aae03e8e419bcbc44c5544237.svg?branch=main)](https://buildkite.com/julialang/luxcore-dot-jl)
[![CI](https://github.com/LuxDL/LuxCore.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/LuxDL/LuxCore.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/LuxDL/LuxCore.jl/branch/main/graph/badge.svg?token=1ZY0A2NPEM)](https://codecov.io/gh/LuxDL/LuxCore.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

> [!WARNING]
> Package has been moved to a subdirectory in Lux https://github.com/LuxDL/Lux.jl/tree/main/lib/

`LuxCore.jl` defines the abstract layers for Lux. Allows users to be compatible with the
entirely of `Lux.jl` without having such a heavy dependency. If you are depending on
`Lux.jl` directly, you do not need to depend on `LuxCore.jl` (all the functionality is
exported via `Lux.jl`).
