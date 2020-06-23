# nginx_config
添加用户名和密码进行登录

# 参考网页

http://www.tashan10.com/nginxshe-zhi-wang-zhan-fang-wen-mi-ma/

# 下载htpasswd工具

apt-get install apache2-utils

# 添加用户名和密码

htpasswd -bdc passwd admin 123456

passwd就是用户名和密码的文件，在conf中的auth_basic_user_file项进行加载，用户名/密码 是 admin/123456
