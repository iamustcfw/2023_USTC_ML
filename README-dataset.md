## 检测社交媒体虚假账户

数据集中有三个文件
- train.json
- test.json
- rawtest.json: 不要修改这个文件, 用来检测test.json的格式是否规范

目标: 预测test.json中的label为 "human" / "bot"

每条数据包含多个字段, 字段说明如下：

- `created_at`: 用户注册日期和时间。
- `user`: 用户的基本信息。
- `id`: 用户的独特标识号码。
- `id_str`: 用户ID的字符串形式。
- `name`: 用户的显示名字，一般为真实姓名或昵称。
- `screen_name`: 用户在平台上的唯一昵称。
- `location`: 用户自行填写的所在地信息。
- `description`: 用户自我描述，简述个人信息。
- `url`: 用户提供的个人或商业网站链接。
- `entities`: 包含用户描述和网站链接的详细信息。
- `protected`: 用户账号的隐私保护设置。
- `followers_count`: 关注该用户的人数。
- `friends_count`: 该用户所关注的人数。
- `listed_count`: 用户被列入他人列表的次数，反映了受欢迎程度。
- `favourites_count`: 用户标记喜欢的推文或帖子数。
- `utc_offset`: 用户所在时区的偏移量。
- `time_zone`: 用户所处的时区。
- `geo_enabled`: 用户是否开启地理位置服务。
- `verified`: 用户是否经过官方认证。
- `statuses_count`: 用户发布的总推文或帖子数。
- `lang`: 用户账号设置的语言。
- `contributors_enabled`: 用户是否开启贡献者模式。
- `is_translator`: 用户是否担任翻译员角色。
- `is_translation_enabled`: 用户是否开启翻译服务。
- `profile_background_color`: 用户资料的背景颜色。
- `profile_background_image_url`: 用户资料背景的图片链接。
- `profile_background_image_url_https`: 用户资料背景图片的安全链接。
- `profile_background_tile`: 用户资料背景是否为平铺式。
- `profile_image_url`: 用户资料的头像链接。
- `profile_image_url_https`: 用户资料头像的安全链接。
- `profile_banner_url`: 用户资料的横幅图片链接。
- `profile_link_color`: 用户资料中链接的颜色。
- `profile_sidebar_border_color`: 用户资料侧栏的边框颜色。
- `profile_sidebar_fill_color`: 用户资料侧栏的填充颜色。
- `profile_text_color`: 用户资料文本的颜色。
- `profile_use_background_image`: 用户是否使用资料背景图片。
- `has_extended_profile`: 用户是否开启扩展资料。
- `default_profile`: 用户是否使用默认资料设置。
- `default_profile_image`: 用户是否使用默认头像。
- `following`: 当前用户是否关注该用户。
- `follow_request_sent`: 当前用户是否向该用户发送关注请求。
- `notifications`: 当前用户是否接收来自该用户的通知。
- `translator_type`: 用户的类型，如翻译员等。
- `label`: 账号的标签，标识为“human”（真实用户）或“bot”（机器人账号）。



上面的代码将依次进行无用特征删除、字符串和布尔型数据处理、对数预处理、特征挖掘、特征过滤等特征工程处理方法，最终得到32个特征：

`done`name_fuzz_ratio: name与screen_name的相似度, 分箱为0-4
description_length: 用户的个人描述的长度
description_polarity: 用户的个人描述的正负面性
description_subjectivity: 用户的个人描述的主客观性
`done`description_url_number: 用户的个人描述中含有的url数目
`done`'url: 用户是否提供网站链接, 非空为1，否则为0 
`done`followers_count: 关注该用户的人数
`done`friends_count: 用户关注的其他用户数
`done`listed_count: 用户被其他用户列入列表的次数
`done`created_at_week: 用户账号创建时间所在星期
`done`created_at_hour: 用户账号创建时间所在小时
`done`favourites_count: 用户标记为喜欢的推文数量
`done`geo_enabled: 用户是否启用地理位置信息
`done`verified: 是否为认证账号
`done`statuses_count: 用户发布的推文总数量
`done`lang: 用户账号的语言设置, 分箱为0-17
profile_background_image_url: 用户的个人资料背景图片
`done`profile_background_tile: 用户的个人资料背景是否平铺
`done`profile_use_background_image: 用户是否使用个人资料背景图片
`done`has_extended_profile: 用户是否启用了扩展个人资料
`done`default_profile: 用户是否使用默认个人资料
`done`translator_type: 用户类型(是否为翻译员)
mined_feature_[0-9]: SymbolicTransformer挖掘出的10个特征






