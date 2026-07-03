# cqaip.cn Site Mapping Report

> 调研日期: 2026-06-22

---

## 一、站点身份

| 项目 | 内容 |
|------|------|
| 站点名称 | **重庆市人工智能公共服务平台** (Chongqing AI Public Service Platform) |
| 域名 | `cqaip.cn` |
| 备案号 | 渝ICP备2025077048号-2 |
| 定位 | 面向AI创业者的公益服务型社区，整合供需、算力、模型、政策等资源 |
| Slogan | "从 IDEA 到 IPO" |

## 二、技术架构

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Next.js 14+ (App Router) | 客户端渲染(CSR)，BailoutToCSR 模式 |
| UI框架 | Ant Design 5.x | 中文 locale |
| 后端 API | NestJS / Express | `/api/` 路由到独立后端服务 |
| 服务器 | nginx 反向代理 | 前端 + 后端统一域名 |
| 状态管理 | React Context | 主题、布局、站点配置 |
| 子域名 | `dataservice.cqaip.cn`, `agents.cqaip.cn` | 数据工厂和智能体工坊独立部署 |

## 三、页面结构

### 主要路由（全部返回200，Next.js CSR 渲染）

| 路由 | 内容 | 数据形态 |
|------|------|---------|
| `/` | 首页（轮播图+统计+网格卡片） | 站点配置 |
| `/policies` | **政策中心** | ⭐ 可能有政策FAQ |
| `/docs` | **文档中心** | ⭐ 可能有帮助文档 |
| `/user-manual` | **用户手册** | ⭐ 可能有FAQ |
| `/community` | **产业风向/社区** | ⭐ 活动+资讯列表 |
| `/community/home` | **OPC社区** | 社区动态 |
| `/industry-news` | **行业资讯** | 新闻列表 |
| `/industry/dynamics` | **行业动态** | 动态列表 |
| `/industry/activities` | **活动信息** | 活动列表 |
| `/industry/alliance` | **产业联盟** | 联盟信息 |
| `/industry/competitions/:id` | **赛事详情** | 富文本详情页 |
| `/enterprise` | **企业风采** | 企业展示 |
| `/marketplace` | **供需大厅** | 供需列表 |
| `/marketplace/demands` | 场景需求大厅 | 需求列表 |
| `/marketplace/products` | 产品方案大厅 | 产品列表 |
| `/marketplace/achievements` | 前沿技术成果 | 成果列表 |
| `/marketplace/benchmarks` | 标杆案例 | 案例列表 |
| `/models` | 模型广场 | 模型列表（需auth） |
| `/compute` | 算力广场 | 算力服务 |
| `/datasets` | 数据集市场 | 数据集 |
| `/skills` | 技能商店 | 技能列表 |
| `/apps` | 应用 | 应用列表 |
| `/ai-agent` | 智能体工坊 | Agent列表 |
| `/enterprise-service` | 企业服务 | 服务介绍 |
| `/opc-communities` | OPC社区列表 | 社区列表 |

## 四、API 接口

### 已知公共接口

| 方法 | 端点 | 状态 | 返回 |
|------|------|------|------|
| GET | `/api/site-config` | ✅ 200 | 站点配置（JSON） |
| GET | `/api/v1/models?page=1&limit=10` | ⚠️ 401 | 需认证 |
| POST | `/api/public/policy-chat` | ❌ 404 | 前端引用但后端未部署 |

### API 响应格式

```json
{
  "code": 200,
  "message": "success",
  "data": { ... },
  "timestamp": "2026-06-22T09:46:50.684Z"
}
```

### 站点配置中的关键数据（来自 /api/site-config）

- **轮播图**: 4 张（阿里 OPC 社区、项目征集、建邺 AI、创客大赛）
- **热点赛事**: 2 个（"满天星"挑战赛、数龙杯AI大赛）
- **统计数据**: 用户189、工具89、方案111、Token 3.1M
- **入驻企业**: 中国信通院、中国移动、中国联通、中国电信、华为云、阿里云

## 五、数据源分析（QA 相关）

### 可能的 Q&A 对来源

| 优先级 | 来源 | 内容类型 | 提取方式 |
|--------|------|---------|---------|
| ⭐⭐⭐ | `/policies` 政策中心 | 政策解读FAQ | 页面RSC数据 |
| ⭐⭐⭐ | `/docs` 文档中心 | 操作文档/FAQ | 页面RSC数据 |
| ⭐⭐⭐ | `/user-manual` 用户手册 | 使用指南/FAQ | 页面RSC数据 |
| ⭐⭐ | `/community` 社区 | 帖子+回复 | 页面RSC数据 |
| ⭐⭐ | `/industry-news` 行业资讯 | 新闻文章 | 页面RSC数据 |
| ⭐ | `/marketplace/benchmarks` 标杆案例 | 案例文章 | 页面RSC数据 |

### 关键发现

1. **没有传统静态 HTML 内容**——所有页面数据通过 Next.js RSC Payload 或客户端 fetch 加载
2. **内容 API 未直接暴露**——公共数据 API 端点被隐藏在前端路由之后
3. **子域名存在**——`dataservice.cqaip.cn` 和 `agents.cqaip.cn` 可能有独立的数据接口

## 六、爬取策略建议

### 方案 A: Playwright 浏览器渲染（推荐）

```
Playwright → 打开页面 → 等待渲染 → 拦截 Network Response → 提取数据
```

- ✅ 能获取完整渲染内容
- ✅ 可拦截 API 响应直接获得结构化数据
- ❌ 较慢，资源消耗大
- ❌ 需要处理反爬（如有）

### 方案 B: Next.js RSC 协议

```
GET /policies + Header: RSC: 1 → 获取 RSC Payload → 解析二进制数据
```

- ✅ 直接获取数据，不需要渲染
- ❌ RSC 格式是 React 内部协议，解析复杂

### 方案 C: 直接爬 + 分析 JS 中的 API 调用

```
下载各页面 JS chunk → 提取 API 端点 → 直接调 API
```

- ✅ 最快
- ❌ 需要大量逆向工程

### 推荐：方案 A（Playwright）

对于 FAQ/QA 类数据，建议先用 Playwright 打开 `/policies`、`/docs`、`/user-manual` 三个最可能的页面，观察网络请求找到实际 API，然后直接调 API。

## 七、反爬评估

| 指标 | 状态 |
|------|------|
| robots.txt | ❌ 无（返回404） |
| 验证码 | 未知（前端有 captcha 配置，当前未启用） |
| 频率限制 | 未检测到（需实际爬取验证） |
| Cookie/Token | `/api/v1/models` 返回401，部分接口需认证 |
| 公开页面 | 全部可访问，无登录墙 |

## 八、数据量估算

| 数据类别 | 预估量级 |
|----------|---------|
| 政策文档 | 10-100 篇 |
| 帮助文档 | 10-100 篇 |
| 行业资讯 | 100-1000 篇 |
| 社区帖子 | 未知 |
| 企业信息 | 6-50 家 |
| 赛事活动 | 10-50 个 |

总体数据量偏小（估计几百到几千条内容），不存在存储压力。
