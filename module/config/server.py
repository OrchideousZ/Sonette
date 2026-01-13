"""
This file stores server, such as 'cn', 'en'.
Use 'import module.config.server as server' to import, don't use 'from xxx import xxx'.
"""
lang = 'cn'  # Setting default to cn, will avoid errors when using dev_tools
server = 'CN-Official'

VALID_LANG = ['cn', 'en']

# === 修改点 1：把重返未来1999加入到 VALID_SERVER ===
VALID_SERVER = {
    'CN-Official': 'com.miHoYo.hkrpg',
    'CN-Bilibili': 'com.miHoYo.hkrpg.bilibili',
    'OVERSEA-America': 'com.HoYoverse.hkrpgoversea',
    'OVERSEA-Asia': 'com.HoYoverse.hkrpgoversea',
    'OVERSEA-Europe': 'com.HoYoverse.hkrpgoversea',
    'OVERSEA-TWHKMO': 'com.HoYoverse.hkrpgoversea',

    # 新增1999配置 (Key可以自定义，Value必须是包名)
    'CN-Reverse1999': 'com.shenlan.m.reverse1999',
}

# 这一行会自动把新加的包名更新进去，不用动
VALID_PACKAGE = set(list(VALID_SERVER.values()))

VALID_CLOUD_SERVER = {
    'CN-Official': 'com.miHoYo.cloudgames.hkrpg',
}
VALID_CLOUD_PACKAGE = set(list(VALID_CLOUD_SERVER.values()))

# === 修改点 2：确认 Activity 配置 ===
DICT_PACKAGE_TO_ACTIVITY = {
    'com.miHoYo.hkrpg': 'com.mihoyo.combosdk.ComboSDKActivity',
    'com.miHoYo.hkrpg.bilibili': 'com.mihoyo.combosdk.ComboSDKActivity',
    'com.HoYoverse.hkrpgoversea': 'com.mihoyo.combosdk.ComboSDKActivity',
    'com.miHoYo.cloudgames.hkrpg': 'com.mihoyo.cloudgame.ui.SplashActivity',

    # 1999配置
    #'com.shenlan.m.reverse1999': 'com.shenlan.m.reverse1999.GamePlayerActivity',
    'com.shenlan.m.reverse1999': 'com.ssgame.mobile.gamesdk.frame.AppStartUpActivity',
}


def set_lang(lang_: str):
    """
    Change language and this will affect globally,
    including assets and language specific methods.

    Args:
        lang_: package name or server.
    """
    global lang
    lang = lang_

    from module.base.resource import release_resources
    release_resources()


def to_server(package_or_server: str) -> str:
    """
    Convert package/server to server.
    To unknown packages, consider they are a CN channel servers.
    """
    # Can't distinguish different regions of oversea servers,
    # assume it's 'OVERSEA-Asia'
    if package_or_server == 'com.HoYoverse.hkrpgoversea':
        return 'OVERSEA-Asia'

    for key, value in VALID_SERVER.items():
        if value == package_or_server:
            return key
        if key == package_or_server:
            return key
    for key, value in VALID_CLOUD_SERVER.items():
        if value == package_or_server:
            return key
        if key == package_or_server:
            return key

    # 之前报错就是因为在这里没找到包名，加上 VALID_SERVER 后就不会报错了
    raise ValueError(f'Package invalid: {package_or_server}')


def to_package(package_or_server: str, is_cloud=False) -> str:
    """
    Convert package/server to package.
    """
    if is_cloud:
        for key, value in VALID_CLOUD_SERVER.items():
            if value == package_or_server:
                return value
            if key == package_or_server:
                return value
    else:
        for key, value in VALID_SERVER.items():
            if value == package_or_server:
                return value
            if key == package_or_server:
                return value

    raise ValueError(f'Server invalid: {package_or_server}')