-- carRAG MySQL 建库脚本（仅建库）
--   挂在 mysql 容器的 /docker-entrypoint-initdb.d/，首次启动镜像自动执行。
--   只负责"建好 subjects_kg 空库"；装数据（建表 jpkb + 灌 9755 条）由独立脚本
--   `python mysql_qa/db/mysql_client.py --force` 完成 —— 见 Docker 数据初始化路径。
-- 对齐 config.ini [mysql]：database=subjects_kg；root 密码由 compose 的
--   MYSQL_ROOT_PASSWORD 注入，这里不碰 root 账号。
CREATE DATABASE IF NOT EXISTS `subjects_kg`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;