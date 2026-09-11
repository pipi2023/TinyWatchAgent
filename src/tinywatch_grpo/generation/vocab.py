"""Real well-known films used as a CPU fixture when the full IMDb dump is absent.

Ratings, years, runtimes and directors are public IMDb facts for these titles.
The production catalog is built from official IMDb non-commercial TSVs.
"""

from __future__ import annotations

GENRE_EN_TO_ZH = {
    "Action": "动作",
    "Adventure": "冒险",
    "Animation": "动画",
    "Biography": "传记",
    "Comedy": "喜剧",
    "Crime": "犯罪",
    "Documentary": "纪录片",
    "Drama": "剧情",
    "Family": "家庭",
    "Fantasy": "奇幻",
    "Film-Noir": "黑色电影",
    "History": "历史",
    "Horror": "恐怖",
    "Music": "音乐",
    "Musical": "歌舞",
    "Mystery": "悬疑",
    "Romance": "爱情",
    "Sci-Fi": "科幻",
    "Sport": "运动",
    "Thriller": "惊悚",
    "War": "战争",
    "Western": "西部",
}

QUERY_TEMPLATES = {
    "easy": (
        "帮我选{n}部{genres}片，总片长不超过{runtime}分钟，评分至少{rating}，{extra}。",
        "想看{genres}，{n}部就行，总时长≤{runtime}分钟，IMDb≥{rating}。{extra}",
    ),
    "medium": (
        "准备一个{n}部片单：类型覆盖{genres}，年份{year_span}，总片长不超过{runtime}分钟，评分≥{rating}。{extra}",
        "周末片单{n}部，要有{genres}，{year_span}，合计片长≤{runtime}分钟，不要低分（≥{rating}）。{extra}",
    ),
    "hard": (
        "严格约束：{n}部、类型{genres}、{year_span}、总片长≤{runtime}分钟、评分≥{rating}。{extra}找不到就 abort，不要硬凑。",
        "硬条件同时满足：部数{n}、{genres}、{year_span}、时长预算{runtime}分钟、IMDb≥{rating}。{extra}",
    ),
}

# Frozen real titles for tests / smoke. Production data comes from IMDb dumps.
MINI_MOVIES = [
    {"movie_id": "tt0111161", "title_zh": "肖申克的救赎", "title_en": "The Shawshank Redemption", "year": 1994, "runtime": 142, "rating": 9.3, "votes": 2900000, "genres": ["剧情"], "directors": ["Frank Darabont"], "has_zh_aka": True},
    {"movie_id": "tt0068646", "title_zh": "教父", "title_en": "The Godfather", "year": 1972, "runtime": 175, "rating": 9.2, "votes": 2000000, "genres": ["犯罪", "剧情"], "directors": ["Francis Ford Coppola"], "has_zh_aka": True},
    {"movie_id": "tt0468569", "title_zh": "蝙蝠侠：黑暗骑士", "title_en": "The Dark Knight", "year": 2008, "runtime": 152, "rating": 9.0, "votes": 2900000, "genres": ["动作", "犯罪", "剧情"], "directors": ["Christopher Nolan"], "has_zh_aka": True},
    {"movie_id": "tt1375666", "title_zh": "盗梦空间", "title_en": "Inception", "year": 2010, "runtime": 148, "rating": 8.8, "votes": 2500000, "genres": ["动作", "科幻", "惊悚"], "directors": ["Christopher Nolan"], "has_zh_aka": True},
    {"movie_id": "tt0816692", "title_zh": "星际穿越", "title_en": "Interstellar", "year": 2014, "runtime": 169, "rating": 8.7, "votes": 2100000, "genres": ["冒险", "剧情", "科幻"], "directors": ["Christopher Nolan"], "has_zh_aka": True},
    {"movie_id": "tt0482571", "title_zh": "致命魔术", "title_en": "The Prestige", "year": 2006, "runtime": 130, "rating": 8.5, "votes": 1400000, "genres": ["剧情", "悬疑", "科幻"], "directors": ["Christopher Nolan"], "has_zh_aka": True},
    {"movie_id": "tt0209144", "title_zh": "记忆碎片", "title_en": "Memento", "year": 2000, "runtime": 113, "rating": 8.4, "votes": 1300000, "genres": ["悬疑", "惊悚"], "directors": ["Christopher Nolan"], "has_zh_aka": True},
    {"movie_id": "tt15398776", "title_zh": "奥本海默", "title_en": "Oppenheimer", "year": 2023, "runtime": 180, "rating": 8.3, "votes": 800000, "genres": ["传记", "剧情", "历史"], "directors": ["Christopher Nolan"], "has_zh_aka": True},
    {"movie_id": "tt0137523", "title_zh": "搏击俱乐部", "title_en": "Fight Club", "year": 1999, "runtime": 139, "rating": 8.8, "votes": 2300000, "genres": ["剧情"], "directors": ["David Fincher"], "has_zh_aka": True},
    {"movie_id": "tt0114369", "title_zh": "七宗罪", "title_en": "Se7en", "year": 1995, "runtime": 127, "rating": 8.6, "votes": 1800000, "genres": ["犯罪", "剧情", "悬疑"], "directors": ["David Fincher"], "has_zh_aka": True},
    {"movie_id": "tt1285016", "title_zh": "社交网络", "title_en": "The Social Network", "year": 2010, "runtime": 120, "rating": 7.8, "votes": 750000, "genres": ["传记", "剧情"], "directors": ["David Fincher"], "has_zh_aka": True},
    {"movie_id": "tt2267998", "title_zh": "消失的爱人", "title_en": "Gone Girl", "year": 2014, "runtime": 149, "rating": 8.1, "votes": 1100000, "genres": ["剧情", "悬疑", "惊悚"], "directors": ["David Fincher"], "has_zh_aka": True},
    {"movie_id": "tt0110912", "title_zh": "低俗小说", "title_en": "Pulp Fiction", "year": 1994, "runtime": 154, "rating": 8.9, "votes": 2200000, "genres": ["犯罪", "剧情"], "directors": ["Quentin Tarantino"], "has_zh_aka": True},
    {"movie_id": "tt0361748", "title_zh": "无耻混蛋", "title_en": "Inglourious Basterds", "year": 2009, "runtime": 153, "rating": 8.4, "votes": 1500000, "genres": ["冒险", "剧情", "战争"], "directors": ["Quentin Tarantino"], "has_zh_aka": True},
    {"movie_id": "tt1853728", "title_zh": "被解救的姜戈", "title_en": "Django Unchained", "year": 2012, "runtime": 165, "rating": 8.5, "votes": 1700000, "genres": ["剧情", "西部"], "directors": ["Quentin Tarantino"], "has_zh_aka": True},
    {"movie_id": "tt0133093", "title_zh": "黑客帝国", "title_en": "The Matrix", "year": 1999, "runtime": 136, "rating": 8.7, "votes": 2000000, "genres": ["动作", "科幻"], "directors": ["Lana Wachowski", "Lilly Wachowski"], "has_zh_aka": True},
    {"movie_id": "tt0167260", "title_zh": "指环王：王者归来", "title_en": "The Lord of the Rings: The Return of the King", "year": 2003, "runtime": 201, "rating": 9.0, "votes": 2000000, "genres": ["冒险", "剧情", "奇幻"], "directors": ["Peter Jackson"], "has_zh_aka": True},
    {"movie_id": "tt0120737", "title_zh": "指环王：护戒使者", "title_en": "The Lord of the Rings: The Fellowship of the Ring", "year": 2001, "runtime": 178, "rating": 8.9, "votes": 2000000, "genres": ["冒险", "剧情", "奇幻"], "directors": ["Peter Jackson"], "has_zh_aka": True},
    {"movie_id": "tt0109830", "title_zh": "阿甘正传", "title_en": "Forrest Gump", "year": 1994, "runtime": 142, "rating": 8.8, "votes": 2200000, "genres": ["剧情", "爱情"], "directors": ["Robert Zemeckis"], "has_zh_aka": True},
    {"movie_id": "tt0120338", "title_zh": "泰坦尼克号", "title_en": "Titanic", "year": 1997, "runtime": 194, "rating": 7.9, "votes": 1300000, "genres": ["剧情", "爱情"], "directors": ["James Cameron"], "has_zh_aka": True},
    {"movie_id": "tt0499549", "title_zh": "阿凡达", "title_en": "Avatar", "year": 2009, "runtime": 162, "rating": 7.9, "votes": 1400000, "genres": ["动作", "冒险", "科幻"], "directors": ["James Cameron"], "has_zh_aka": True},
    {"movie_id": "tt0080684", "title_zh": "星球大战：帝国反击战", "title_en": "Star Wars: Episode V - The Empire Strikes Back", "year": 1980, "runtime": 124, "rating": 8.7, "votes": 1400000, "genres": ["动作", "冒险", "科幻"], "directors": ["Irvin Kershner"], "has_zh_aka": True},
    {"movie_id": "tt0088763", "title_zh": "回到未来", "title_en": "Back to the Future", "year": 1985, "runtime": 116, "rating": 8.5, "votes": 1300000, "genres": ["冒险", "喜剧", "科幻"], "directors": ["Robert Zemeckis"], "has_zh_aka": True},
    {"movie_id": "tt0081505", "title_zh": "闪灵", "title_en": "The Shining", "year": 1980, "runtime": 146, "rating": 8.4, "votes": 1100000, "genres": ["剧情", "恐怖"], "directors": ["Stanley Kubrick"], "has_zh_aka": True},
    {"movie_id": "tt0062622", "title_zh": "2001太空漫游", "title_en": "2001: A Space Odyssey", "year": 1968, "runtime": 149, "rating": 8.3, "votes": 700000, "genres": ["冒险", "科幻"], "directors": ["Stanley Kubrick"], "has_zh_aka": True},
    {"movie_id": "tt1160419", "title_zh": "沙丘", "title_en": "Dune", "year": 2021, "runtime": 155, "rating": 8.0, "votes": 900000, "genres": ["动作", "冒险", "科幻"], "directors": ["Denis Villeneuve"], "has_zh_aka": True},
    {"movie_id": "tt2543164", "title_zh": "降临", "title_en": "Arrival", "year": 2016, "runtime": 116, "rating": 7.9, "votes": 750000, "genres": ["剧情", "科幻", "惊悚"], "directors": ["Denis Villeneuve"], "has_zh_aka": True},
    {"movie_id": "tt1856101", "title_zh": "银翼杀手2049", "title_en": "Blade Runner 2049", "year": 2017, "runtime": 164, "rating": 8.0, "votes": 650000, "genres": ["动作", "剧情", "科幻"], "directors": ["Denis Villeneuve"], "has_zh_aka": True},
    {"movie_id": "tt6751668", "title_zh": "寄生虫", "title_en": "Parasite", "year": 2019, "runtime": 132, "rating": 8.5, "votes": 1000000, "genres": ["剧情", "惊悚"], "directors": ["Bong Joon Ho"], "has_zh_aka": True},
    {"movie_id": "tt0245429", "title_zh": "千与千寻", "title_en": "Spirited Away", "year": 2001, "runtime": 125, "rating": 8.6, "votes": 900000, "genres": ["动画", "冒险", "家庭"], "directors": ["Hayao Miyazaki"], "has_zh_aka": True},
    {"movie_id": "tt0347149", "title_zh": "哈尔的移动城堡", "title_en": "Howl's Moving Castle", "year": 2004, "runtime": 119, "rating": 8.2, "votes": 450000, "genres": ["动画", "冒险", "家庭"], "directors": ["Hayao Miyazaki"], "has_zh_aka": True},
    {"movie_id": "tt5311514", "title_zh": "你的名字。", "title_en": "Your Name.", "year": 2016, "runtime": 106, "rating": 8.4, "votes": 350000, "genres": ["动画", "剧情", "奇幻"], "directors": ["Makoto Shinkai"], "has_zh_aka": True},
    {"movie_id": "tt0118694", "title_zh": "花样年华", "title_en": "In the Mood for Love", "year": 2000, "runtime": 98, "rating": 8.1, "votes": 170000, "genres": ["剧情", "爱情"], "directors": ["Kar-Wai Wong"], "has_zh_aka": True},
    {"movie_id": "tt0109508", "title_zh": "重庆森林", "title_en": "Chungking Express", "year": 1994, "runtime": 102, "rating": 8.0, "votes": 100000, "genres": ["喜剧", "犯罪", "剧情"], "directors": ["Kar-Wai Wong"], "has_zh_aka": True},
    {"movie_id": "tt0107156", "title_zh": "霸王别姬", "title_en": "Farewell My Concubine", "year": 1993, "runtime": 171, "rating": 8.1, "votes": 35000, "genres": ["剧情", "爱情"], "directors": ["Kaige Chen"], "has_zh_aka": True},
    {"movie_id": "tt0446059", "title_zh": "功夫", "title_en": "Kung Fu Hustle", "year": 2004, "runtime": 99, "rating": 7.7, "votes": 150000, "genres": ["动作", "喜剧", "奇幻"], "directors": ["Stephen Chow"], "has_zh_aka": True},
    {"movie_id": "tt0299977", "title_zh": "卧虎藏龙", "title_en": "Crouching Tiger, Hidden Dragon", "year": 2000, "runtime": 120, "rating": 7.9, "votes": 280000, "genres": ["动作", "冒险", "剧情"], "directors": ["Ang Lee"], "has_zh_aka": True},
    {"movie_id": "tt0298203", "title_zh": "无间道", "title_en": "Infernal Affairs", "year": 2002, "runtime": 101, "rating": 8.0, "votes": 130000, "genres": ["动作", "犯罪", "剧情"], "directors": ["Andrew Lau", "Alan Mak"], "has_zh_aka": True},
    {"movie_id": "tt1305806", "title_zh": "让子弹飞", "title_en": "Let the Bullets Fly", "year": 2010, "runtime": 132, "rating": 7.3, "votes": 20000, "genres": ["动作", "喜剧"], "directors": ["Wen Jiang"], "has_zh_aka": True},
    {"movie_id": "tt4154796", "title_zh": "复仇者联盟4：终局之战", "title_en": "Avengers: Endgame", "year": 2019, "runtime": 181, "rating": 8.4, "votes": 1300000, "genres": ["动作", "冒险", "科幻"], "directors": ["Anthony Russo", "Joe Russo"], "has_zh_aka": True},
    {"movie_id": "tt1745960", "title_zh": "壮志凌云2：独行侠", "title_en": "Top Gun: Maverick", "year": 2022, "runtime": 130, "rating": 8.2, "votes": 700000, "genres": ["动作", "剧情"], "directors": ["Joseph Kosinski"], "has_zh_aka": True},
    {"movie_id": "tt1517268", "title_zh": "芭比", "title_en": "Barbie", "year": 2023, "runtime": 114, "rating": 6.8, "votes": 500000, "genres": ["冒险", "喜剧", "奇幻"], "directors": ["Greta Gerwig"], "has_zh_aka": True},
    {"movie_id": "tt2380307", "title_zh": "寻梦环游记", "title_en": "Coco", "year": 2017, "runtime": 105, "rating": 8.4, "votes": 600000, "genres": ["动画", "冒险", "家庭"], "directors": ["Lee Unkrich", "Adrian Molina"], "has_zh_aka": True},
    {"movie_id": "tt0910970", "title_zh": "机器人总动员", "title_en": "WALL·E", "year": 2008, "runtime": 98, "rating": 8.4, "votes": 1200000, "genres": ["动画", "冒险", "家庭"], "directors": ["Andrew Stanton"], "has_zh_aka": True},
    {"movie_id": "tt0114709", "title_zh": "玩具总动员", "title_en": "Toy Story", "year": 1995, "runtime": 81, "rating": 8.3, "votes": 1100000, "genres": ["动画", "冒险", "喜剧"], "directors": ["John Lasseter"], "has_zh_aka": True},
    {"movie_id": "tt2582802", "title_zh": "爆裂鼓手", "title_en": "Whiplash", "year": 2014, "runtime": 106, "rating": 8.5, "votes": 1000000, "genres": ["剧情", "音乐"], "directors": ["Damien Chazelle"], "has_zh_aka": True},
    {"movie_id": "tt2278388", "title_zh": "布达佩斯大饭店", "title_en": "The Grand Budapest Hotel", "year": 2014, "runtime": 99, "rating": 8.1, "votes": 900000, "genres": ["冒险", "喜剧", "犯罪"], "directors": ["Wes Anderson"], "has_zh_aka": True},
    {"movie_id": "tt1392190", "title_zh": "疯狂的麦克斯：狂暴之路", "title_en": "Mad Max: Fury Road", "year": 2015, "runtime": 120, "rating": 8.1, "votes": 1100000, "genres": ["动作", "冒险", "科幻"], "directors": ["George Miller"], "has_zh_aka": True},
    {"movie_id": "tt0108052", "title_zh": "辛德勒的名单", "title_en": "Schindler's List", "year": 1993, "runtime": 195, "rating": 9.0, "votes": 1400000, "genres": ["传记", "剧情", "历史"], "directors": ["Steven Spielberg"], "has_zh_aka": True},
    {"movie_id": "tt0120815", "title_zh": "拯救大兵瑞恩", "title_en": "Saving Private Ryan", "year": 1998, "runtime": 169, "rating": 8.6, "votes": 1500000, "genres": ["剧情", "战争"], "directors": ["Steven Spielberg"], "has_zh_aka": True},
]


def mini_catalog() -> dict:
    movies = []
    for row in MINI_MOVIES:
        movie = dict(row)
        movie["original_title"] = movie["title_en"]
        movie["akas"] = [movie["title_zh"], movie["title_en"]]
        movies.append(movie)
    return {
        "schema_version": "tinywatch-catalog-v1",
        "source": "imdb-mini-fixture",
        "attribution": "IMDb non-commercial dataset facts; Chinese titles from public akas.",
        "movies": movies,
    }
