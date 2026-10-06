# 飞书机器人平台配置清单

> 目标：让群里 @ 机器人 → 传文件入库 / 提问回答 成为可能。
> 本文档只描述可选飞书平台配置；当前单实例部署见 ../deploy/standalone.md。
> 前置：有权使用的飞书应用，凭据仅写当前项目私有 .env。本次不启用机器人。

## 1. 应用加"机器人"能力

飞书开放平台 → 应用详情 → **添加应用能力 → 机器人** → 启用。

## 2. 权限（scope）

应用权限管理里开通以下权限（机器人相关）：

| 权限 | 用途 |
|---|---|
| `im:message` | 接收群消息事件（识别 @ 与消息内容） |
| `im:message:send_as_bot` | 机器人向会话发送回执/答案 |
| `im:resource` | 下载用户发送的文件（入库用） |

开通后需要**创建版本并发布**（企业内自建应用：发布后管理员/企业内生效；可用"测试企业"验证）。

## 3. 事件订阅：长连接模式

事件与回调 → **订阅方式选"长连接"**（不配置回调 URL，应用主动出站连飞书，无需公网入口）→ 添加事件 **`im.message.receive_v1`**（接收消息）。

## 4. 把机器人拉进群

发布后：通讯录/群设置 → 添加机器人（或应用可用范围包含目标群成员）。

## 5. 验证

- 群里 @ 机器人发一条消息 → 应收到机器人"收到，处理中"回执（需另行运行机器人进程）。
- 非目标租户/群的 @ 不响应（代码侧过滤，见环境配置）。

## 环境配置（当前私有 .env 追加）

```ini
# 复用 OAuth 的 app id/secret（同一应用）
FEISHU_APP_ID=<已有值>
FEISHU_APP_SECRET=<已有值>
FEISHU_ALLOWED_TENANT=<已有值>      # 租户限定：非此租户事件忽略
FEISHU_BOT_ENABLED=true             # 机器人进程开关
FEISHU_BOT_ALLOWED_CHAT_IDS=        # 群白名单（逗号分隔 chat_id，留空=租户内不限）
FEISHU_BOT_INBOX_PROJECT=feishu-inbox  # 文件入库项目（项目不存在自动创建）
FEISHU_BOT_API_BASE_URL=http://127.0.0.1:18002  # 当前 loopback API
```

另行加载同一私有配置，用 API 环境执行 `python -m app.services.feishu_bot.entry`。
机器人不归 project_ctl 管理，没有默认崩溃自动拉起；不复制历史公司的凭据。
