const merchants = [
  {
    id: "m1",
    name: "红炉川味冒菜",
    category: "川湘辣味",
    cuisine: "川菜",
    tastes: ["麻辣", "重口味", "热食"],
    scenes: ["午餐", "晚餐", "加班"],
    area: "中关村",
    price: 34,
    delivery: 3.5,
    distance: 1.2,
    rating: 4.8,
    monthly: 2480,
    foods: ["牛肉冒菜", "午餐套餐"],
    image: "./assets/sichuan.svg",
    accent: "#f56a4d",
  },
  {
    id: "m2",
    name: "岭南烧腊饭堂",
    category: "盖饭便当",
    cuisine: "粤式",
    tastes: ["咸鲜", "肉食", "热食"],
    scenes: ["午餐", "工作餐"],
    area: "五道口",
    price: 31,
    delivery: 2,
    distance: 0.9,
    rating: 4.7,
    monthly: 1920,
    foods: ["烧鸭饭", "叉烧拼鸡"],
    image: "./assets/rice.svg",
    accent: "#f4b63f",
  },
  {
    id: "m3",
    name: "青柠轻食实验室",
    category: "轻食沙拉",
    cuisine: "健康餐",
    tastes: ["清淡", "低脂", "蔬菜"],
    scenes: ["午餐", "健身", "下午"],
    area: "中关村",
    price: 38,
    delivery: 4,
    distance: 1.7,
    rating: 4.9,
    monthly: 1360,
    foods: ["鸡胸沙拉", "藜麦碗"],
    image: "./assets/salad.svg",
    accent: "#3fb984",
  },
  {
    id: "m4",
    name: "夜航炸鸡汉堡",
    category: "炸鸡汉堡",
    cuisine: "西式快餐",
    tastes: ["香脆", "肉食", "高热量"],
    scenes: ["夜宵", "周末", "加班"],
    area: "望京",
    price: 42,
    delivery: 5,
    distance: 2.4,
    rating: 4.6,
    monthly: 2210,
    foods: ["辣翅桶", "芝士汉堡"],
    image: "./assets/chicken.svg",
    accent: "#c667a6",
  },
  {
    id: "m5",
    name: "潮汕鲜牛丸粿条",
    category: "粉面汤粥",
    cuisine: "潮汕",
    tastes: ["鲜香", "汤面", "热食"],
    scenes: ["早餐", "午餐", "晚餐"],
    area: "五道口",
    price: 29,
    delivery: 2.5,
    distance: 1.0,
    rating: 4.8,
    monthly: 1670,
    foods: ["牛丸粿条", "牛肉汤"],
    image: "./assets/noodles.svg",
    accent: "#519be8",
  },
  {
    id: "m6",
    name: "朴素东北家常菜",
    category: "家常小炒",
    cuisine: "东北菜",
    tastes: ["咸鲜", "下饭", "大份量"],
    scenes: ["晚餐", "多人餐"],
    area: "西二旗",
    price: 45,
    delivery: 3,
    distance: 2.1,
    rating: 4.5,
    monthly: 960,
    foods: ["锅包肉", "地三鲜"],
    image: "./assets/stirfry.svg",
    accent: "#f56a4d",
  },
  {
    id: "m7",
    name: "桂花糖水铺",
    category: "甜品饮品",
    cuisine: "广式甜品",
    tastes: ["甜口", "冰饮", "小食"],
    scenes: ["下午茶", "夜宵"],
    area: "中关村",
    price: 24,
    delivery: 2,
    distance: 0.7,
    rating: 4.9,
    monthly: 3020,
    foods: ["杨枝甘露", "双皮奶"],
    image: "./assets/dessert.svg",
    accent: "#c667a6",
  },
  {
    id: "m8",
    name: "秦巷肉夹馍凉皮",
    category: "地方小吃",
    cuisine: "西北",
    tastes: ["酸辣", "面食", "小吃"],
    scenes: ["午餐", "快餐", "夜宵"],
    area: "西二旗",
    price: 23,
    delivery: 1.5,
    distance: 1.4,
    rating: 4.6,
    monthly: 1740,
    foods: ["肉夹馍", "秦镇凉皮"],
    image: "./assets/snack.svg",
    accent: "#f4b63f",
  },
  {
    id: "m9",
    name: "番茄牛腩米线",
    category: "粉面汤粥",
    cuisine: "云南米线",
    tastes: ["酸甜", "汤面", "热食"],
    scenes: ["午餐", "晚餐", "雨天"],
    area: "望京",
    price: 32,
    delivery: 3,
    distance: 1.8,
    rating: 4.7,
    monthly: 1430,
    foods: ["番茄牛腩米线", "菌菇米线"],
    image: "./assets/noodles.svg",
    accent: "#f56a4d",
  },
  {
    id: "m10",
    name: "一盏粥铺",
    category: "早餐粥点",
    cuisine: "粥点",
    tastes: ["清淡", "暖胃", "热食"],
    scenes: ["早餐", "病中餐", "夜宵"],
    area: "五道口",
    price: 21,
    delivery: 1,
    distance: 0.6,
    rating: 4.7,
    monthly: 1190,
    foods: ["皮蛋瘦肉粥", "虾饺"],
    image: "./assets/porridge.svg",
    accent: "#3fb984",
  },
  {
    id: "m11",
    name: "韩式石锅拌饭",
    category: "日韩料理",
    cuisine: "韩式",
    tastes: ["微辣", "拌饭", "蔬菜"],
    scenes: ["午餐", "工作餐"],
    area: "中关村",
    price: 36,
    delivery: 3.5,
    distance: 1.5,
    rating: 4.6,
    monthly: 1320,
    foods: ["牛肉拌饭", "泡菜汤"],
    image: "./assets/rice.svg",
    accent: "#519be8",
  },
  {
    id: "m12",
    name: "沪上小笼生煎",
    category: "包子点心",
    cuisine: "江浙",
    tastes: ["咸甜", "小吃", "热食"],
    scenes: ["早餐", "午餐", "快餐"],
    area: "西二旗",
    price: 27,
    delivery: 2,
    distance: 1.1,
    rating: 4.5,
    monthly: 1850,
    foods: ["小笼包", "鲜肉生煎"],
    image: "./assets/dim-sum.svg",
    accent: "#f4b63f",
  },
  {
    id: "m13",
    name: "椰香海南鸡饭",
    category: "盖饭便当",
    cuisine: "海南菜",
    tastes: ["清爽", "鸡肉", "热食"],
    scenes: ["午餐", "工作餐"],
    area: "望京",
    price: 35,
    delivery: 2.5,
    distance: 1.6,
    rating: 4.8,
    monthly: 1510,
    foods: ["海南鸡饭", "冬瓜汤"],
    image: "./assets/rice.svg",
    accent: "#3fb984",
  },
  {
    id: "m14",
    name: "炉火披萨意面",
    category: "披萨意面",
    cuisine: "意式",
    tastes: ["芝士", "番茄", "高热量"],
    scenes: ["晚餐", "周末", "多人餐"],
    area: "中关村",
    price: 58,
    delivery: 6,
    distance: 2.9,
    rating: 4.6,
    monthly: 870,
    foods: ["玛格丽特披萨", "肉酱意面"],
    image: "./assets/pizza.svg",
    accent: "#f56a4d",
  },
  {
    id: "m15",
    name: "鲜切水果酸奶杯",
    category: "水果酸奶",
    cuisine: "健康甜品",
    tastes: ["甜口", "清爽", "低脂"],
    scenes: ["下午茶", "健身", "夜宵"],
    area: "五道口",
    price: 26,
    delivery: 2,
    distance: 0.8,
    rating: 4.8,
    monthly: 2140,
    foods: ["酸奶水果杯", "芒果盒子"],
    image: "./assets/fruit.svg",
    accent: "#c667a6",
  },
  {
    id: "m16",
    name: "云贵酸汤鱼",
    category: "特色正餐",
    cuisine: "贵州菜",
    tastes: ["酸辣", "鱼鲜", "热食"],
    scenes: ["晚餐", "多人餐", "雨天"],
    area: "西二旗",
    price: 52,
    delivery: 4.5,
    distance: 2.5,
    rating: 4.7,
    monthly: 780,
    foods: ["酸汤鱼", "折耳根拌菜"],
    image: "./assets/fish.svg",
    accent: "#519be8",
  },
];

const extraMerchants = [
  {
    id: "m17",
    name: "湘味剁椒蒸菜",
    category: "川湘辣味",
    cuisine: "湘菜",
    tastes: ["香辣", "蒜香", "热食"],
    scenes: ["晚餐", "下饭", "多人餐"],
    area: "五道口",
    price: 39,
    delivery: 3,
    distance: 1.3,
    rating: 4.7,
    monthly: 1680,
    foods: ["剁椒鸡腿饭", "蒸腊味双拼"],
    ingredients: ["剁椒", "鸡腿肉", "金针菇"],
    cooking: ["蒸菜", "现炒"],
    nutrition: ["高蛋白"],
    timeSlots: ["晚高峰"],
    crowd: ["重口味人群", "合餐人群"],
    portions: ["单人套餐", "双人餐"],
    benefits: ["满减"],
    deliveryTags: ["出餐稳定"],
    serviceTags: ["包装严实"],
    image: "./assets/sichuan.svg",
    accent: "#f56a4d",
  },
  {
    id: "m18",
    name: "田园鸡胸能量碗",
    category: "轻食沙拉",
    cuisine: "健康餐",
    tastes: ["清淡", "低脂", "蔬菜"],
    scenes: ["午餐", "健身", "工作餐"],
    area: "望京",
    price: 41,
    delivery: 4,
    distance: 1.9,
    rating: 4.8,
    monthly: 1260,
    foods: ["香煎鸡胸能量碗", "牛油果虾仁沙拉"],
    ingredients: ["鸡胸肉", "牛油果", "藜麦", "生菜"],
    cooking: ["煎烤", "冷食"],
    nutrition: ["低脂", "高蛋白", "控糖"],
    timeSlots: ["午高峰", "下午"],
    crowd: ["健身人群", "轻食人群"],
    portions: ["单人套餐"],
    benefits: ["新客券"],
    deliveryTags: ["冷链包装"],
    serviceTags: ["酱料分装"],
    image: "./assets/salad.svg",
    accent: "#3fb984",
  },
  {
    id: "m19",
    name: "深夜炭火烤串",
    category: "烧烤夜宵",
    cuisine: "东北烧烤",
    tastes: ["孜然", "香辣", "肉食"],
    scenes: ["夜宵", "周末", "多人餐"],
    area: "西二旗",
    price: 49,
    delivery: 5,
    distance: 2.6,
    rating: 4.5,
    monthly: 2380,
    foods: ["羊肉串", "烤鸡翅", "烤韭菜"],
    ingredients: ["羊肉", "鸡翅", "韭菜"],
    cooking: ["炭烤", "现烤"],
    nutrition: ["高蛋白", "高热量"],
    timeSlots: ["夜间"],
    crowd: ["夜宵人群", "合餐人群"],
    portions: ["多人分享"],
    benefits: ["满减"],
    deliveryTags: ["夜间配送"],
    serviceTags: ["保温包装"],
    image: "./assets/chicken.svg",
    accent: "#c667a6",
  },
  {
    id: "m20",
    name: "一人食麻辣烫",
    category: "麻辣烫冒菜",
    cuisine: "川味",
    tastes: ["麻辣", "自选", "热食"],
    scenes: ["午餐", "晚餐", "加班"],
    area: "中关村",
    price: 28,
    delivery: 2.5,
    distance: 0.8,
    rating: 4.6,
    monthly: 2860,
    foods: ["牛油麻辣烫", "番茄麻辣烫"],
    ingredients: ["牛肉丸", "豆皮", "青菜", "宽粉"],
    cooking: ["煮制", "自选"],
    nutrition: ["可加蔬菜"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["重口味人群", "选择困难"],
    portions: ["单人套餐"],
    benefits: ["满减", "免配送费"],
    deliveryTags: ["近距离"],
    serviceTags: ["汤底可选"],
    image: "./assets/sichuan.svg",
    accent: "#f56a4d",
  },
  {
    id: "m21",
    name: "兰州清汤牛肉面",
    category: "粉面汤粥",
    cuisine: "西北",
    tastes: ["清汤", "面食", "热食"],
    scenes: ["早餐", "午餐", "快餐"],
    area: "西二旗",
    price: 24,
    delivery: 1.5,
    distance: 0.9,
    rating: 4.7,
    monthly: 2110,
    foods: ["牛肉拉面", "凉拌牛肉"],
    ingredients: ["牛肉", "拉面", "萝卜"],
    cooking: ["现煮", "手工面"],
    nutrition: ["高蛋白", "暖胃"],
    timeSlots: ["早餐", "午高峰"],
    crowd: ["快餐人群", "面食偏好"],
    portions: ["单人套餐"],
    benefits: ["折扣套餐"],
    deliveryTags: ["出餐快"],
    serviceTags: ["汤面分装"],
    image: "./assets/noodles.svg",
    accent: "#519be8",
  },
  {
    id: "m22",
    name: "广式早茶点心局",
    category: "包子点心",
    cuisine: "粤式",
    tastes: ["咸鲜", "小吃", "热食"],
    scenes: ["早餐", "下午茶", "多人餐"],
    area: "五道口",
    price: 46,
    delivery: 3.5,
    distance: 1.7,
    rating: 4.8,
    monthly: 1470,
    foods: ["虾饺皇", "豉汁凤爪", "流沙包"],
    ingredients: ["虾仁", "猪肉", "蛋黄"],
    cooking: ["蒸点", "手工点心"],
    nutrition: ["小份多样"],
    timeSlots: ["早餐", "下午"],
    crowd: ["点心偏好", "合餐人群"],
    portions: ["多人分享", "小份尝鲜"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["餐具齐全"],
    image: "./assets/dim-sum.svg",
    accent: "#f4b63f",
  },
  {
    id: "m23",
    name: "阿嬷卤肉饭",
    category: "盖饭便当",
    cuisine: "台湾",
    tastes: ["酱香", "肉食", "下饭"],
    scenes: ["午餐", "工作餐", "晚餐"],
    area: "中关村",
    price: 33,
    delivery: 2,
    distance: 0.7,
    rating: 4.7,
    monthly: 2240,
    foods: ["台式卤肉饭", "盐酥鸡饭"],
    ingredients: ["猪肉", "鸡腿肉", "卤蛋"],
    cooking: ["卤制", "炸制"],
    nutrition: ["高蛋白", "高热量"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["工作餐人群", "肉食偏好"],
    portions: ["单人套餐", "加量饭"],
    benefits: ["满减"],
    deliveryTags: ["出餐快"],
    serviceTags: ["米饭可加"],
    image: "./assets/rice.svg",
    accent: "#f4b63f",
  },
  {
    id: "m24",
    name: "越南牛肉河粉",
    category: "东南亚料理",
    cuisine: "越南",
    tastes: ["清爽", "鲜香", "汤面"],
    scenes: ["午餐", "晚餐", "雨天"],
    area: "望京",
    price: 37,
    delivery: 3,
    distance: 1.5,
    rating: 4.6,
    monthly: 940,
    foods: ["生牛肉河粉", "柠檬鸡丝檬粉"],
    ingredients: ["牛肉", "河粉", "香草", "柠檬"],
    cooking: ["现煮", "汤粉"],
    nutrition: ["清爽", "高蛋白"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["汤粉偏好", "清淡人群"],
    portions: ["单人套餐"],
    benefits: ["新客券"],
    deliveryTags: ["汤粉分装"],
    serviceTags: ["配料分装"],
    image: "./assets/noodles.svg",
    accent: "#3fb984",
  },
  {
    id: "m25",
    name: "曼谷黄咖喱鸡",
    category: "东南亚料理",
    cuisine: "泰式",
    tastes: ["咖喱", "微辣", "椰香"],
    scenes: ["午餐", "晚餐", "工作餐"],
    area: "五道口",
    price: 43,
    delivery: 4,
    distance: 2.0,
    rating: 4.6,
    monthly: 860,
    foods: ["黄咖喱鸡饭", "冬阴功汤"],
    ingredients: ["鸡腿肉", "咖喱", "椰浆"],
    cooking: ["炖煮", "盖饭"],
    nutrition: ["高蛋白"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["异国料理偏好"],
    portions: ["单人套餐"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["辣度可选"],
    image: "./assets/rice.svg",
    accent: "#f4b63f",
  },
  {
    id: "m26",
    name: "筑地寿司便当",
    category: "日韩料理",
    cuisine: "日式",
    tastes: ["清爽", "海鲜", "冷食"],
    scenes: ["午餐", "下午", "轻食"],
    area: "中关村",
    price: 55,
    delivery: 5,
    distance: 2.2,
    rating: 4.7,
    monthly: 760,
    foods: ["三文鱼寿司", "鳗鱼饭"],
    ingredients: ["三文鱼", "海苔", "米饭"],
    cooking: ["冷食", "刺身"],
    nutrition: ["高蛋白", "低脂"],
    timeSlots: ["午高峰", "下午"],
    crowd: ["海鲜偏好", "轻食人群"],
    portions: ["单人套餐", "小份尝鲜"],
    benefits: ["会员券"],
    deliveryTags: ["冷链包装"],
    serviceTags: ["芥末酱油分装"],
    image: "./assets/rice.svg",
    accent: "#519be8",
  },
  {
    id: "m27",
    name: "韩式炸酱年糕屋",
    category: "日韩料理",
    cuisine: "韩式",
    tastes: ["甜辣", "芝士", "小吃"],
    scenes: ["下午茶", "夜宵", "周末"],
    area: "望京",
    price: 35,
    delivery: 3.5,
    distance: 1.4,
    rating: 4.5,
    monthly: 1530,
    foods: ["芝士年糕", "韩式炸酱面"],
    ingredients: ["年糕", "芝士", "泡菜"],
    cooking: ["炒制", "炸制"],
    nutrition: ["高热量"],
    timeSlots: ["下午", "夜间"],
    crowd: ["甜辣偏好", "小吃偏好"],
    portions: ["双人餐", "小份尝鲜"],
    benefits: ["满减"],
    deliveryTags: ["夜间配送"],
    serviceTags: ["辣度可选"],
    image: "./assets/snack.svg",
    accent: "#c667a6",
  },
  {
    id: "m28",
    name: "新疆大盘鸡拌面",
    category: "地方小吃",
    cuisine: "新疆",
    tastes: ["香辣", "面食", "大份量"],
    scenes: ["晚餐", "多人餐", "加班"],
    area: "西二旗",
    price: 48,
    delivery: 4,
    distance: 2.3,
    rating: 4.6,
    monthly: 1320,
    foods: ["大盘鸡拌面", "羊肉抓饭"],
    ingredients: ["鸡肉", "土豆", "宽面"],
    cooking: ["炖煮", "拌面"],
    nutrition: ["高蛋白", "高热量"],
    timeSlots: ["晚高峰"],
    crowd: ["面食偏好", "合餐人群"],
    portions: ["多人分享", "加量饭"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["大份量"],
    image: "./assets/noodles.svg",
    accent: "#f56a4d",
  },
  {
    id: "m29",
    name: "柳州螺蛳粉铺",
    category: "粉面汤粥",
    cuisine: "广西",
    tastes: ["酸辣", "重口味", "汤粉"],
    scenes: ["午餐", "夜宵", "雨天"],
    area: "五道口",
    price: 30,
    delivery: 2.5,
    distance: 1.0,
    rating: 4.5,
    monthly: 2560,
    foods: ["经典螺蛳粉", "炸蛋螺蛳粉"],
    ingredients: ["米粉", "酸笋", "腐竹", "炸蛋"],
    cooking: ["现煮", "汤粉"],
    nutrition: ["重口味"],
    timeSlots: ["午高峰", "夜间"],
    crowd: ["重口味人群", "粉面偏好"],
    portions: ["单人套餐", "加料"],
    benefits: ["满减"],
    deliveryTags: ["汤粉分装"],
    serviceTags: ["配料分装"],
    image: "./assets/noodles.svg",
    accent: "#f56a4d",
  },
  {
    id: "m30",
    name: "砂锅土豆粉",
    category: "粉面汤粥",
    cuisine: "河南",
    tastes: ["酸辣", "暖胃", "热食"],
    scenes: ["晚餐", "雨天", "加班"],
    area: "中关村",
    price: 27,
    delivery: 2,
    distance: 0.9,
    rating: 4.6,
    monthly: 1810,
    foods: ["砂锅土豆粉", "番茄肥牛粉"],
    ingredients: ["土豆粉", "肥牛", "青菜"],
    cooking: ["砂锅", "现煮"],
    nutrition: ["暖胃", "可加蔬菜"],
    timeSlots: ["晚高峰", "夜间"],
    crowd: ["暖胃偏好", "粉面偏好"],
    portions: ["单人套餐"],
    benefits: ["免配送费"],
    deliveryTags: ["近距离", "汤粉分装"],
    serviceTags: ["保温包装"],
    image: "./assets/noodles.svg",
    accent: "#519be8",
  },
  {
    id: "m31",
    name: "卤味拌饭研究所",
    category: "盖饭便当",
    cuisine: "家常卤味",
    tastes: ["酱香", "咸鲜", "下饭"],
    scenes: ["午餐", "工作餐", "晚餐"],
    area: "西二旗",
    price: 32,
    delivery: 2,
    distance: 0.8,
    rating: 4.7,
    monthly: 1990,
    foods: ["卤鸡腿饭", "卤肉拼盘饭"],
    ingredients: ["鸡腿肉", "猪肉", "卤蛋"],
    cooking: ["卤制", "盖饭"],
    nutrition: ["高蛋白"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["工作餐人群", "肉食偏好"],
    portions: ["单人套餐", "加量饭"],
    benefits: ["满减"],
    deliveryTags: ["出餐快"],
    serviceTags: ["米饭可加"],
    image: "./assets/rice.svg",
    accent: "#f4b63f",
  },
  {
    id: "m32",
    name: "素食星球便当",
    category: "素食轻食",
    cuisine: "健康餐",
    tastes: ["清淡", "蔬菜", "谷物"],
    scenes: ["午餐", "健身", "工作餐"],
    area: "望京",
    price: 36,
    delivery: 3,
    distance: 1.2,
    rating: 4.8,
    monthly: 840,
    foods: ["菌菇豆腐饭", "鹰嘴豆蔬菜碗"],
    ingredients: ["豆腐", "菌菇", "鹰嘴豆", "糙米"],
    cooking: ["蒸煮", "少油"],
    nutrition: ["低脂", "高纤维", "素食"],
    timeSlots: ["午高峰", "下午"],
    crowd: ["素食人群", "轻食人群"],
    portions: ["单人套餐"],
    benefits: ["新客券"],
    deliveryTags: ["低油包装"],
    serviceTags: ["热量标注"],
    image: "./assets/salad.svg",
    accent: "#3fb984",
  },
  {
    id: "m33",
    name: "云南野菌焖饭",
    category: "盖饭便当",
    cuisine: "云南菜",
    tastes: ["鲜香", "菌菇", "热食"],
    scenes: ["午餐", "晚餐", "工作餐"],
    area: "中关村",
    price: 40,
    delivery: 3.5,
    distance: 1.6,
    rating: 4.7,
    monthly: 920,
    foods: ["野菌鸡肉焖饭", "菌菇汽锅汤"],
    ingredients: ["菌菇", "鸡肉", "米饭"],
    cooking: ["焖饭", "炖汤"],
    nutrition: ["高蛋白", "菌菇"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["菌菇偏好", "工作餐人群"],
    portions: ["单人套餐"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["汤饭分装"],
    image: "./assets/rice.svg",
    accent: "#3fb984",
  },
  {
    id: "m34",
    name: "热辣重庆小面",
    category: "粉面汤粥",
    cuisine: "重庆",
    tastes: ["麻辣", "面食", "重口味"],
    scenes: ["早餐", "午餐", "加班"],
    area: "五道口",
    price: 22,
    delivery: 1.5,
    distance: 0.5,
    rating: 4.6,
    monthly: 2450,
    foods: ["豌杂小面", "肥肠小面"],
    ingredients: ["面条", "豌豆", "肥肠"],
    cooking: ["现煮", "拌面"],
    nutrition: ["暖胃"],
    timeSlots: ["早餐", "午高峰"],
    crowd: ["重口味人群", "面食偏好"],
    portions: ["单人套餐"],
    benefits: ["折扣套餐"],
    deliveryTags: ["出餐快", "近距离"],
    serviceTags: ["辣度可选"],
    image: "./assets/noodles.svg",
    accent: "#f56a4d",
  },
  {
    id: "m35",
    name: "海盐芝士奶茶",
    category: "甜品饮品",
    cuisine: "新茶饮",
    tastes: ["甜口", "奶香", "冰饮"],
    scenes: ["下午茶", "周末", "加班"],
    area: "中关村",
    price: 19,
    delivery: 1.5,
    distance: 0.4,
    rating: 4.8,
    monthly: 3820,
    foods: ["海盐芝士奶茶", "多肉葡萄"],
    ingredients: ["茶", "牛奶", "芝士", "水果"],
    cooking: ["现制饮品", "冷饮"],
    nutrition: ["高糖", "冰饮"],
    timeSlots: ["下午", "夜间"],
    crowd: ["甜品偏好", "饮品偏好"],
    portions: ["单杯", "双杯套餐"],
    benefits: ["第二杯半价"],
    deliveryTags: ["近距离"],
    serviceTags: ["少糖可选"],
    image: "./assets/dessert.svg",
    accent: "#c667a6",
  },
  {
    id: "m36",
    name: "精品咖啡轻食站",
    category: "咖啡轻食",
    cuisine: "咖啡简餐",
    tastes: ["咖啡", "清爽", "低脂"],
    scenes: ["早餐", "下午茶", "加班"],
    area: "望京",
    price: 33,
    delivery: 2.5,
    distance: 0.9,
    rating: 4.7,
    monthly: 1670,
    foods: ["拿铁", "火鸡三明治"],
    ingredients: ["咖啡豆", "牛奶", "全麦面包", "火鸡肉"],
    cooking: ["现制饮品", "烘烤"],
    nutrition: ["低脂", "轻食"],
    timeSlots: ["早餐", "下午"],
    crowd: ["咖啡偏好", "轻食人群"],
    portions: ["单人套餐"],
    benefits: ["早餐套餐"],
    deliveryTags: ["出餐快"],
    serviceTags: ["冷热可选"],
    image: "./assets/fruit.svg",
    accent: "#519be8",
  },
  {
    id: "m37",
    name: "酸奶燕麦水果仓",
    category: "水果酸奶",
    cuisine: "健康甜品",
    tastes: ["甜口", "清爽", "低脂"],
    scenes: ["早餐", "下午茶", "健身"],
    area: "西二旗",
    price: 28,
    delivery: 2,
    distance: 0.7,
    rating: 4.8,
    monthly: 1760,
    foods: ["希腊酸奶碗", "草莓燕麦杯"],
    ingredients: ["酸奶", "燕麦", "草莓", "蓝莓"],
    cooking: ["冷食", "现切"],
    nutrition: ["低脂", "高纤维", "控糖"],
    timeSlots: ["早餐", "下午"],
    crowd: ["健身人群", "甜品偏好"],
    portions: ["单杯", "小份尝鲜"],
    benefits: ["会员券"],
    deliveryTags: ["冷链包装"],
    serviceTags: ["糖度可选"],
    image: "./assets/fruit.svg",
    accent: "#c667a6",
  },
  {
    id: "m38",
    name: "热卤鸭货小馆",
    category: "卤味小吃",
    cuisine: "武汉卤味",
    tastes: ["香辣", "酱香", "小吃"],
    scenes: ["夜宵", "下午茶", "周末"],
    area: "五道口",
    price: 34,
    delivery: 3,
    distance: 1.5,
    rating: 4.5,
    monthly: 1880,
    foods: ["鸭脖鸭锁骨", "卤藕片"],
    ingredients: ["鸭肉", "藕片", "豆干"],
    cooking: ["卤制", "凉拌"],
    nutrition: ["高蛋白", "重口味"],
    timeSlots: ["下午", "夜间"],
    crowd: ["小吃偏好", "夜宵人群"],
    portions: ["多人分享", "小份尝鲜"],
    benefits: ["满减"],
    deliveryTags: ["夜间配送"],
    serviceTags: ["真空包装"],
    image: "./assets/snack.svg",
    accent: "#f56a4d",
  },
  {
    id: "m39",
    name: "鲜虾云吞竹升面",
    category: "粉面汤粥",
    cuisine: "粤式",
    tastes: ["鲜香", "汤面", "清淡"],
    scenes: ["早餐", "午餐", "雨天"],
    area: "中关村",
    price: 35,
    delivery: 2.5,
    distance: 1.1,
    rating: 4.8,
    monthly: 1340,
    foods: ["鲜虾云吞面", "净云吞"],
    ingredients: ["虾仁", "竹升面", "青菜"],
    cooking: ["现煮", "手工面"],
    nutrition: ["高蛋白", "暖胃"],
    timeSlots: ["早餐", "午高峰"],
    crowd: ["清淡人群", "汤面偏好"],
    portions: ["单人套餐"],
    benefits: ["折扣套餐"],
    deliveryTags: ["汤面分装"],
    serviceTags: ["汤底分装"],
    image: "./assets/noodles.svg",
    accent: "#519be8",
  },
  {
    id: "m40",
    name: "潮汕砂锅粥王",
    category: "早餐粥点",
    cuisine: "潮汕",
    tastes: ["鲜香", "暖胃", "清淡"],
    scenes: ["早餐", "晚餐", "病中餐"],
    area: "望京",
    price: 44,
    delivery: 4,
    distance: 2.1,
    rating: 4.7,
    monthly: 980,
    foods: ["鲜虾砂锅粥", "干贝排骨粥"],
    ingredients: ["鲜虾", "大米", "干贝", "排骨"],
    cooking: ["砂锅", "慢煮"],
    nutrition: ["暖胃", "高蛋白"],
    timeSlots: ["早餐", "晚高峰"],
    crowd: ["暖胃偏好", "清淡人群"],
    portions: ["双人餐", "多人分享"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["粥料分装"],
    image: "./assets/porridge.svg",
    accent: "#3fb984",
  },
  {
    id: "m41",
    name: "江南小炒黄牛肉",
    category: "家常小炒",
    cuisine: "湘赣菜",
    tastes: ["香辣", "下饭", "热食"],
    scenes: ["午餐", "晚餐", "多人餐"],
    area: "西二旗",
    price: 47,
    delivery: 4,
    distance: 2.0,
    rating: 4.6,
    monthly: 1190,
    foods: ["小炒黄牛肉", "辣椒炒肉"],
    ingredients: ["黄牛肉", "辣椒", "蒜苗"],
    cooking: ["现炒", "小炒"],
    nutrition: ["高蛋白"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["重口味人群", "合餐人群"],
    portions: ["双人餐", "多人分享"],
    benefits: ["满减"],
    deliveryTags: ["保温包装"],
    serviceTags: ["米饭可加"],
    image: "./assets/stirfry.svg",
    accent: "#f56a4d",
  },
  {
    id: "m42",
    name: "铁板黑椒牛排饭",
    category: "盖饭便当",
    cuisine: "西式简餐",
    tastes: ["黑椒", "肉食", "热食"],
    scenes: ["午餐", "晚餐", "工作餐"],
    area: "五道口",
    price: 44,
    delivery: 3.5,
    distance: 1.8,
    rating: 4.6,
    monthly: 1040,
    foods: ["黑椒牛排饭", "香煎鸡排饭"],
    ingredients: ["牛肉", "鸡排", "玉米"],
    cooking: ["铁板", "煎烤"],
    nutrition: ["高蛋白", "高热量"],
    timeSlots: ["午高峰", "晚高峰"],
    crowd: ["肉食偏好", "工作餐人群"],
    portions: ["单人套餐", "加量饭"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["酱汁分装"],
    image: "./assets/rice.svg",
    accent: "#c667a6",
  },
  {
    id: "m43",
    name: "香草烤鱼饭",
    category: "特色正餐",
    cuisine: "江湖菜",
    tastes: ["麻辣", "鱼鲜", "热食"],
    scenes: ["晚餐", "多人餐", "周末"],
    area: "中关村",
    price: 56,
    delivery: 5,
    distance: 2.5,
    rating: 4.7,
    monthly: 870,
    foods: ["香辣烤鱼饭", "豆花烤鱼"],
    ingredients: ["鱼肉", "豆花", "香菜"],
    cooking: ["烤制", "炖煮"],
    nutrition: ["高蛋白"],
    timeSlots: ["晚高峰"],
    crowd: ["鱼鲜偏好", "合餐人群"],
    portions: ["双人餐", "多人分享"],
    benefits: ["满减"],
    deliveryTags: ["保温包装"],
    serviceTags: ["汤汁防漏"],
    image: "./assets/fish.svg",
    accent: "#519be8",
  },
  {
    id: "m44",
    name: "妈妈手作水饺",
    category: "包子点心",
    cuisine: "北方家常",
    tastes: ["咸鲜", "面食", "热食"],
    scenes: ["晚餐", "病中餐", "家庭餐"],
    area: "望京",
    price: 29,
    delivery: 2.5,
    distance: 1.3,
    rating: 4.7,
    monthly: 1420,
    foods: ["三鲜水饺", "猪肉白菜饺"],
    ingredients: ["猪肉", "虾仁", "韭菜", "面粉"],
    cooking: ["水煮", "手工点心"],
    nutrition: ["暖胃", "高蛋白"],
    timeSlots: ["晚高峰"],
    crowd: ["家庭餐人群", "面食偏好"],
    portions: ["单人套餐", "家庭份"],
    benefits: ["折扣套餐"],
    deliveryTags: ["保温包装"],
    serviceTags: ["蘸料分装"],
    image: "./assets/dim-sum.svg",
    accent: "#f4b63f",
  },
  {
    id: "m45",
    name: "闽南沙茶拌面",
    category: "地方小吃",
    cuisine: "闽南",
    tastes: ["沙茶", "鲜香", "面食"],
    scenes: ["早餐", "午餐", "快餐"],
    area: "西二旗",
    price: 26,
    delivery: 2,
    distance: 0.9,
    rating: 4.6,
    monthly: 1160,
    foods: ["沙茶拌面", "花生汤"],
    ingredients: ["面条", "沙茶酱", "花生"],
    cooking: ["拌面", "现煮"],
    nutrition: ["暖胃"],
    timeSlots: ["早餐", "午高峰"],
    crowd: ["面食偏好", "快餐人群"],
    portions: ["单人套餐"],
    benefits: ["免配送费"],
    deliveryTags: ["近距离"],
    serviceTags: ["酱料分装"],
    image: "./assets/noodles.svg",
    accent: "#f4b63f",
  },
  {
    id: "m46",
    name: "法棍三明治小店",
    category: "咖啡轻食",
    cuisine: "法式简餐",
    tastes: ["清爽", "芝士", "轻食"],
    scenes: ["早餐", "午餐", "下午茶"],
    area: "中关村",
    price: 34,
    delivery: 2.5,
    distance: 1.0,
    rating: 4.7,
    monthly: 930,
    foods: ["火腿芝士法棍", "金枪鱼三明治"],
    ingredients: ["法棍", "火腿", "芝士", "生菜"],
    cooking: ["烘烤", "冷食"],
    nutrition: ["轻食", "高蛋白"],
    timeSlots: ["早餐", "下午"],
    crowd: ["轻食人群", "咖啡偏好"],
    portions: ["单人套餐"],
    benefits: ["早餐套餐"],
    deliveryTags: ["出餐快"],
    serviceTags: ["冷热可选"],
    image: "./assets/snack.svg",
    accent: "#3fb984",
  },
  {
    id: "m47",
    name: "鲜炖银耳桃胶",
    category: "甜品饮品",
    cuisine: "中式甜品",
    tastes: ["甜口", "暖胃", "清爽"],
    scenes: ["下午茶", "夜宵", "病中餐"],
    area: "五道口",
    price: 25,
    delivery: 2,
    distance: 0.8,
    rating: 4.8,
    monthly: 1380,
    foods: ["桃胶银耳羹", "红枣莲子羹"],
    ingredients: ["银耳", "桃胶", "红枣"],
    cooking: ["慢炖", "甜品"],
    nutrition: ["低脂", "暖胃"],
    timeSlots: ["下午", "夜间"],
    crowd: ["甜品偏好", "清淡人群"],
    portions: ["单杯", "双杯套餐"],
    benefits: ["第二杯半价"],
    deliveryTags: ["保温包装"],
    serviceTags: ["甜度可选"],
    image: "./assets/dessert.svg",
    accent: "#c667a6",
  },
  {
    id: "m48",
    name: "家庭装披萨拼盘",
    category: "披萨意面",
    cuisine: "意式",
    tastes: ["芝士", "番茄", "肉食"],
    scenes: ["晚餐", "周末", "多人餐"],
    area: "望京",
    price: 66,
    delivery: 6,
    distance: 3.0,
    rating: 4.5,
    monthly: 980,
    foods: ["超级至尊披萨", "鸡肉凯撒沙拉"],
    ingredients: ["芝士", "番茄", "培根", "鸡肉"],
    cooking: ["烤制", "拼盘"],
    nutrition: ["高热量", "多人分享"],
    timeSlots: ["晚高峰", "夜间"],
    crowd: ["家庭餐人群", "合餐人群"],
    portions: ["家庭份", "多人分享"],
    benefits: ["套餐优惠"],
    deliveryTags: ["保温包装"],
    serviceTags: ["切片配送"],
    image: "./assets/pizza.svg",
    accent: "#f56a4d",
  },
];

const generatedMerchantBlueprints = [
  {
    category: "川湘辣味",
    cuisine: "川菜",
    names: ["椒火水煮肉片", "蜀巷麻婆豆腐", "红油钵钵鸡", "山城辣子鸡", "藤椒牛肉饭", "川府回锅肉"],
    foods: [["水煮肉片", "红油抄手"], ["麻婆豆腐饭", "宫保鸡丁"], ["钵钵鸡", "甜皮鸭"], ["辣子鸡", "酸辣粉"], ["藤椒牛肉饭", "冒菜"], ["回锅肉饭", "鱼香肉丝"]],
    tastes: ["麻辣", "香辣", "重口味", "热食"],
    scenes: ["午餐", "晚餐", "加班"],
    ingredients: ["牛肉", "鸡肉", "豆腐", "辣椒"],
    cooking: ["现炒", "红油", "炖煮"],
    nutrition: ["高蛋白"],
    crowd: ["重口味人群", "工作餐人群"],
    image: "./assets/sichuan.svg",
    accent: "#f56a4d",
  },
  {
    category: "盖饭便当",
    cuisine: "粤式",
    names: ["港记叉烧饭", "广府白切鸡饭", "烧鹅便当屋", "黑椒鸡扒饭", "豉油鸡饭堂", "蜜汁排骨饭"],
    foods: [["叉烧饭", "例汤"], ["白切鸡饭", "冬瓜汤"], ["烧鹅饭", "青菜"], ["黑椒鸡扒饭", "煎蛋"], ["豉油鸡饭", "卤蛋"], ["蜜汁排骨饭", "时蔬"]],
    tastes: ["咸鲜", "肉食", "下饭", "热食"],
    scenes: ["午餐", "工作餐", "晚餐"],
    ingredients: ["鸡肉", "猪肉", "米饭", "青菜"],
    cooking: ["烧腊", "盖饭", "煎烤"],
    nutrition: ["高蛋白"],
    crowd: ["工作餐人群", "肉食偏好"],
    image: "./assets/rice.svg",
    accent: "#f4b63f",
  },
  {
    category: "轻食沙拉",
    cuisine: "健康餐",
    names: ["绿野藜麦碗", "蛋白质沙拉站", "牛油果轻食铺", "低脂鸡胸厨房", "彩虹蔬菜碗", "地中海轻食杯"],
    foods: [["藜麦鸡胸碗", "羽衣甘蓝沙拉"], ["虾仁蛋白沙拉", "酸奶杯"], ["牛油果鸡蛋沙拉", "全麦卷"], ["香煎鸡胸饭", "蔬菜汤"], ["彩虹蔬菜碗", "鹰嘴豆泥"], ["地中海牛肉碗", "番茄汤"]],
    tastes: ["清淡", "低脂", "蔬菜", "清爽"],
    scenes: ["午餐", "健身", "工作餐"],
    ingredients: ["鸡胸肉", "生菜", "藜麦", "牛油果"],
    cooking: ["少油", "冷食", "煎烤"],
    nutrition: ["低脂", "高蛋白", "控糖"],
    crowd: ["健身人群", "轻食人群"],
    image: "./assets/salad.svg",
    accent: "#3fb984",
  },
  {
    category: "粉面汤粥",
    cuisine: "重庆",
    names: ["山城豌杂面", "红汤肥肠面", "番茄肥牛米线", "砂锅酸辣粉", "牛肉汤粉铺", "云吞竹升面"],
    foods: [["豌杂小面", "红油抄手"], ["肥肠面", "凉拌黄瓜"], ["番茄肥牛米线", "菌菇米线"], ["砂锅酸辣粉", "炸蛋"], ["牛肉汤粉", "卤蛋"], ["鲜虾云吞面", "净云吞"]],
    tastes: ["麻辣", "酸辣", "汤面", "热食"],
    scenes: ["早餐", "午餐", "晚餐", "雨天"],
    ingredients: ["面条", "米粉", "牛肉", "青菜"],
    cooking: ["现煮", "汤粉", "手工面"],
    nutrition: ["暖胃", "高蛋白"],
    crowd: ["粉面偏好", "面食偏好"],
    image: "./assets/noodles.svg",
    accent: "#519be8",
  },
  {
    category: "烧烤夜宵",
    cuisine: "东北烧烤",
    names: ["炉边羊肉串", "夜市烤翅局", "炭火牛油小串", "锡纸花甲粉", "烤鱼夜宵档", "东北烤冷面"],
    foods: [["羊肉串", "烤韭菜"], ["辣烤鸡翅", "烤土豆"], ["牛油小串", "烤豆皮"], ["锡纸花甲粉", "烤金针菇"], ["香辣烤鱼", "烤馒头"], ["烤冷面", "炸串"]],
    tastes: ["孜然", "香辣", "肉食", "高热量"],
    scenes: ["夜宵", "周末", "多人餐"],
    ingredients: ["羊肉", "鸡翅", "花甲", "豆皮"],
    cooking: ["炭烤", "现烤", "锡纸"],
    nutrition: ["高蛋白", "高热量"],
    crowd: ["夜宵人群", "合餐人群"],
    image: "./assets/chicken.svg",
    accent: "#c667a6",
  },
  {
    category: "早餐粥点",
    cuisine: "粥点",
    names: ["晨光皮蛋瘦肉粥", "暖胃砂锅粥", "小笼蒸点铺", "鲜虾云吞粥", "南瓜小米粥", "豆浆油条档"],
    foods: [["皮蛋瘦肉粥", "虾饺"], ["鲜虾砂锅粥", "干贝粥"], ["小笼包", "烧麦"], ["云吞粥", "蒸饺"], ["南瓜小米粥", "鸡蛋饼"], ["豆浆", "油条"]],
    tastes: ["清淡", "暖胃", "热食", "咸鲜"],
    scenes: ["早餐", "病中餐", "夜宵"],
    ingredients: ["大米", "虾仁", "瘦肉", "面粉"],
    cooking: ["慢煮", "蒸点", "现煮"],
    nutrition: ["暖胃", "高蛋白"],
    crowd: ["清淡人群", "家庭餐人群"],
    image: "./assets/porridge.svg",
    accent: "#3fb984",
  },
  {
    category: "甜品饮品",
    cuisine: "新茶饮",
    names: ["多肉葡萄茶", "芋泥鲜奶铺", "桂花酒酿圆子", "椰椰水果杯", "芝士莓莓茶", "手作双皮奶"],
    foods: [["多肉葡萄", "芝士奶盖"], ["芋泥鲜奶", "芋圆"], ["酒酿圆子", "银耳羹"], ["椰椰水果杯", "芒果西米露"], ["芝士莓莓", "杨枝甘露"], ["双皮奶", "姜撞奶"]],
    tastes: ["甜口", "奶香", "清爽", "冰饮"],
    scenes: ["下午茶", "夜宵", "周末"],
    ingredients: ["水果", "牛奶", "茶", "芋泥"],
    cooking: ["现制饮品", "冷饮", "甜品"],
    nutrition: ["高糖", "冰饮"],
    crowd: ["甜品偏好", "饮品偏好"],
    image: "./assets/dessert.svg",
    accent: "#c667a6",
  },
  {
    category: "地方小吃",
    cuisine: "西北",
    names: ["秦味油泼面", "潼关肉夹馍", "凉皮擀面皮", "新疆抓饭铺", "沙茶拌面档", "锅盔酸辣粉"],
    foods: [["油泼面", "冰峰"], ["肉夹馍", "凉皮"], ["擀面皮", "肉夹馍"], ["羊肉抓饭", "烤包子"], ["沙茶拌面", "花生汤"], ["锅盔", "酸辣粉"]],
    tastes: ["酸辣", "面食", "小吃", "下饭"],
    scenes: ["午餐", "快餐", "夜宵"],
    ingredients: ["面粉", "牛肉", "猪肉", "辣椒"],
    cooking: ["拌面", "凉拌", "烘烤"],
    nutrition: ["暖胃"],
    crowd: ["面食偏好", "快餐人群"],
    image: "./assets/snack.svg",
    accent: "#f4b63f",
  },
  {
    category: "日韩料理",
    cuisine: "韩式",
    names: ["泡菜石锅饭", "芝士年糕铺", "日式鳗鱼饭", "寿司便当局", "韩式炸鸡屋", "味噌拉面馆"],
    foods: [["石锅拌饭", "泡菜汤"], ["芝士年糕", "炸酱面"], ["鳗鱼饭", "味噌汤"], ["三文鱼寿司", "玉子烧"], ["韩式炸鸡", "年糕"], ["味噌拉面", "叉烧饭"]],
    tastes: ["微辣", "甜辣", "清爽", "芝士"],
    scenes: ["午餐", "下午茶", "夜宵"],
    ingredients: ["米饭", "泡菜", "芝士", "三文鱼"],
    cooking: ["拌饭", "炸制", "冷食"],
    nutrition: ["高蛋白"],
    crowd: ["异国料理偏好", "小吃偏好"],
    image: "./assets/rice.svg",
    accent: "#519be8",
  },
  {
    category: "东南亚料理",
    cuisine: "泰式",
    names: ["泰香咖喱鸡饭", "越南河粉小馆", "椰香冬阴功", "新加坡叻沙铺", "柠檬鸡丝檬粉", "香茅烤肉饭"],
    foods: [["黄咖喱鸡饭", "冬阴功汤"], ["牛肉河粉", "春卷"], ["冬阴功粉", "椰奶冻"], ["叻沙米粉", "沙爹鸡肉"], ["鸡丝檬粉", "柠檬茶"], ["香茅烤肉饭", "青木瓜沙拉"]],
    tastes: ["咖喱", "酸甜", "清爽", "微辣"],
    scenes: ["午餐", "晚餐", "工作餐"],
    ingredients: ["鸡腿肉", "牛肉", "椰浆", "柠檬"],
    cooking: ["炖煮", "汤粉", "烤制"],
    nutrition: ["高蛋白", "清爽"],
    crowd: ["异国料理偏好", "清淡人群"],
    image: "./assets/noodles.svg",
    accent: "#3fb984",
  },
  {
    category: "咖啡轻食",
    cuisine: "咖啡简餐",
    names: ["拿铁三明治站", "法棍咖啡角", "全麦贝果铺", "冷萃轻食仓", "火鸡帕尼尼", "燕麦拿铁小店"],
    foods: [["拿铁", "火鸡三明治"], ["法棍三明治", "美式咖啡"], ["全麦贝果", "酸奶杯"], ["冷萃咖啡", "鸡胸卷"], ["火鸡帕尼尼", "蔬菜汤"], ["燕麦拿铁", "香蕉蛋糕"]],
    tastes: ["咖啡", "清爽", "低脂", "轻食"],
    scenes: ["早餐", "下午茶", "加班"],
    ingredients: ["咖啡豆", "牛奶", "全麦面包", "鸡胸肉"],
    cooking: ["现制饮品", "烘烤", "冷食"],
    nutrition: ["低脂", "轻食"],
    crowd: ["咖啡偏好", "轻食人群"],
    image: "./assets/fruit.svg",
    accent: "#519be8",
  },
  {
    category: "披萨意面",
    cuisine: "意式",
    names: ["番茄肉酱意面", "芝士披萨工坊", "奶油蘑菇面", "家庭披萨拼盘", "香蒜培根意面", "海鲜焗饭屋"],
    foods: [["肉酱意面", "蘑菇汤"], ["玛格丽特披萨", "鸡翅"], ["奶油蘑菇面", "凯撒沙拉"], ["至尊披萨", "薯角"], ["培根意面", "蒜香面包"], ["海鲜焗饭", "番茄汤"]],
    tastes: ["芝士", "番茄", "肉食", "高热量"],
    scenes: ["晚餐", "周末", "多人餐"],
    ingredients: ["芝士", "番茄", "面粉", "培根"],
    cooking: ["烤制", "烘烤", "焗饭"],
    nutrition: ["高热量"],
    crowd: ["家庭餐人群", "合餐人群"],
    image: "./assets/pizza.svg",
    accent: "#f56a4d",
  },
  {
    category: "特色正餐",
    cuisine: "江湖菜",
    names: ["香辣烤鱼饭", "酸汤鱼小馆", "黄焖鸡米饭", "铁锅炖排骨", "番茄牛腩锅", "菌菇鸡汤饭"],
    foods: [["香辣烤鱼", "米饭"], ["酸汤鱼", "折耳根"], ["黄焖鸡米饭", "青菜"], ["铁锅炖排骨", "土豆"], ["番茄牛腩", "米饭"], ["菌菇鸡汤饭", "时蔬"]],
    tastes: ["鱼鲜", "热食", "下饭", "鲜香"],
    scenes: ["晚餐", "多人餐", "雨天"],
    ingredients: ["鱼肉", "鸡肉", "牛肉", "菌菇"],
    cooking: ["炖煮", "烤制", "砂锅"],
    nutrition: ["高蛋白", "暖胃"],
    crowd: ["合餐人群", "鱼鲜偏好"],
    image: "./assets/fish.svg",
    accent: "#519be8",
  },
];

const areas = ["中关村", "五道口", "望京", "西二旗", "国贸", "三里屯", "上地", "回龙观", "亦庄", "朝阳门"];
const benefitOptions = ["满减", "新客券", "折扣套餐", "免配送费", "会员券", "套餐优惠", "第二杯半价"];
const deliveryOptions = ["出餐快", "近距离", "保温包装", "汤面分装", "夜间配送", "冷链包装"];
const serviceOptions = ["包装严实", "辣度可选", "少糖可选", "酱料分装", "餐具齐全", "米饭可加", "配料分装"];
const portionOptions = ["单人套餐", "双人餐", "多人分享", "小份尝鲜", "家庭份", "加量饭"];
const timeSlotOptions = ["早餐", "午高峰", "下午", "晚高峰", "夜间"];
const namePrefixes = ["云上", "巷口", "小满", "拾味", "南城", "北里", "青禾", "满堂", "有间", "烟火", "食光", "邻家", "一味", "米仓", "鲜作", "慢火", "热浪", "晓市"];
const styleWords = ["手作", "鲜煮", "现炒", "匠心", "家常", "风味", "慢炖", "鲜切", "炭火", "砂锅", "热卤", "轻盈"];
const storeSuffixes = ["小馆", "饭堂", "厨房", "食社", "铺", "档", "专门店", "研究所", "便当局", "食堂", "小站", "工坊"];
const suffixByCategory = {
  川湘辣味: ["小馆", "饭堂", "厨房", "食社"],
  盖饭便当: ["饭堂", "便当局", "食堂", "厨房"],
  轻食沙拉: ["轻食站", "沙拉铺", "厨房", "研究所"],
  粉面汤粥: ["面馆", "粉铺", "汤铺", "小馆"],
  烧烤夜宵: ["烤串铺", "夜宵档", "食社", "小馆"],
  早餐粥点: ["粥铺", "点心局", "早餐铺", "小站"],
  甜品饮品: ["甜品铺", "茶饮站", "糖水铺", "小站"],
  地方小吃: ["小吃铺", "面档", "食社", "小馆"],
  日韩料理: ["食堂", "料理屋", "便当局", "小馆"],
  东南亚料理: ["小馆", "食堂", "粉铺", "厨房"],
  咖啡轻食: ["咖啡站", "轻食铺", "小站", "工坊"],
  披萨意面: ["工坊", "披萨铺", "意面屋", "厨房"],
  特色正餐: ["小馆", "饭堂", "厨房", "食社"],
};
const focusByCategory = {
  川湘辣味: ["川味", "辣味", "冒菜", "小炒"],
  盖饭便当: ["热饭", "烧腊", "便当", "盖饭"],
  轻食沙拉: ["轻食", "沙拉", "能量碗", "低脂餐"],
  粉面汤粥: ["汤粉", "鲜汤", "手工面", "砂锅粉"],
  烧烤夜宵: ["烤串", "炉火", "夜宵", "小串"],
  早餐粥点: ["早粥", "蒸点", "早点", "暖粥"],
  甜品饮品: ["甜品", "茶饮", "糖水", "果茶"],
  地方小吃: ["小吃", "面点", "风味", "街巷"],
  日韩料理: ["拌饭", "寿司", "便当", "年糕"],
  东南亚料理: ["咖喱", "河粉", "香茅", "叻沙"],
  咖啡轻食: ["咖啡", "轻食", "贝果", "三明治"],
  披萨意面: ["披萨", "意面", "焗饭", "芝士"],
  特色正餐: ["正餐", "砂锅", "鱼饭", "炖菜"],
};
const styleByCategory = {
  川湘辣味: ["红油", "现炒", "香辣", "家常", "热辣", "小炒"],
  盖饭便当: ["家常", "现烧", "酱香", "热饭", "匠心", "快手"],
  轻食沙拉: ["轻盈", "鲜切", "低脂", "元气", "清爽", "手作"],
  粉面汤粥: ["鲜煮", "砂锅", "暖汤", "手工", "红汤", "清汤"],
  烧烤夜宵: ["炭火", "夜市", "现烤", "孜然", "热辣", "炉边"],
  早餐粥点: ["慢炖", "暖胃", "晨光", "手作", "鲜蒸", "清粥"],
  甜品饮品: ["鲜制", "轻甜", "手作", "冰爽", "桂香", "奶香"],
  地方小吃: ["街巷", "手作", "现拌", "家常", "风味", "热乎"],
  日韩料理: ["炙烧", "手作", "清爽", "酱香", "芝士", "便当"],
  东南亚料理: ["椰香", "香茅", "酸甜", "清爽", "咖喱", "鲜煮"],
  咖啡轻食: ["烘焙", "轻盈", "鲜作", "低脂", "手作", "冷萃"],
  披萨意面: ["烘烤", "芝士", "番茄", "手作", "焗香", "炉火"],
  特色正餐: ["慢炖", "鲜煮", "家常", "砂锅", "现烧", "浓汤"],
};

function pickFrom(list, index, count = 1) {
  return Array.from({ length: count }, (_, offset) => list[(index + offset * 3) % list.length]);
}

function generatedMerchantName(blueprint, index, localIndex, variant) {
  const base = blueprint.names[variant];
  if (localIndex === 0) return base;
  const prefix = namePrefixes[(index + variant * 2 + localIndex) % namePrefixes.length];
  const styles = styleByCategory[blueprint.category] || styleWords;
  const style = styles[(index * 3 + localIndex + variant) % styles.length];
  const suffixes = suffixByCategory[blueprint.category] || storeSuffixes;
  const suffix = suffixes[(index + localIndex * 5 + variant) % suffixes.length];
  const focusWords = focusByCategory[blueprint.category] || [blueprint.category];
  const focus = focusWords[(index + variant + localIndex) % focusWords.length];
  const food = blueprint.foods[(variant + localIndex) % blueprint.foods.length][0].replace(/饭$|面$|粉$|粥$|汤$|茶$|杯$|局$|铺$|档$|馆$|屋$/u, "");
  const cleanBase = base.replace(/铺|屋|馆|档|局|站|角|仓|工坊|小店|小馆|厨房|饭堂|便当屋|夜宵档|专门店/u, "");
  const patterns = [
    `${prefix}${food}${suffix}`,
    `${cleanBase}${style}${suffix}`,
    `${prefix}${style}${focus}${suffix}`,
    `${food}${style}${suffix}`,
  ];
  return patterns[localIndex % patterns.length];
}

function uniquifyGeneratedNames(items) {
  const used = new Map(merchants.map((merchant) => [merchant.name, 1]));
  for (const item of items) {
    const count = used.get(item.name) || 0;
    if (count > 0) {
      const area = item.area || areas[count % areas.length];
      const cuisine = item.cuisine.replace(/菜|餐|料理|简餐/u, "");
      item.name = `${item.name}·${area}${cuisine}`;
    }
    used.set(item.name, (used.get(item.name) || 0) + 1);
  }
  return items;
}

function generatedMerchant(index) {
  const blueprint = generatedMerchantBlueprints[index % generatedMerchantBlueprints.length];
  const localIndex = Math.floor(index / generatedMerchantBlueprints.length);
  const variant = (index + localIndex) % blueprint.names.length;
  const idNumber = merchants.length + extraMerchants.length + index + 1;
  const basePrice = 22 + ((index * 7 + blueprint.category.length * 3) % 43);
  const monthly = 520 + ((index * 137 + variant * 211) % 3420);
  const rating = Number((4.35 + ((index * 17 + variant * 9) % 55) / 100).toFixed(1));
  const distance = Number((0.4 + ((index * 11 + variant * 5) % 31) / 10).toFixed(1));
  const delivery = Number((1 + ((index * 5 + variant * 3) % 11) / 2).toFixed(1));
  return {
    id: `m${idNumber}`,
    name: generatedMerchantName(blueprint, index, localIndex, variant),
    category: blueprint.category,
    cuisine: blueprint.cuisine,
    tastes: unique([...pickFrom(blueprint.tastes, index, 3), blueprint.tastes[variant % blueprint.tastes.length]]),
    scenes: unique([...pickFrom(blueprint.scenes, index, 3), index % 5 === 0 ? "周末" : "工作餐"]),
    area: areas[(index * 2 + variant) % areas.length],
    price: basePrice,
    delivery,
    distance,
    rating,
    monthly,
    foods: blueprint.foods[variant % blueprint.foods.length],
    ingredients: unique([...pickFrom(blueprint.ingredients, index, 3), ...blueprint.ingredients.slice(0, 2)]),
    cooking: unique(pickFrom(blueprint.cooking, index, 2)),
    nutrition: unique([...blueprint.nutrition, index % 4 === 0 ? "可加蔬菜" : "", index % 6 === 0 ? "高纤维" : ""]),
    timeSlots: unique(pickFrom(timeSlotOptions, index + variant, 2)),
    crowd: unique([...blueprint.crowd, index % 3 === 0 ? "工作餐人群" : "", index % 7 === 0 ? "选择困难" : ""]),
    portions: unique(pickFrom(portionOptions, index + variant, 2)),
    benefits: unique(pickFrom(benefitOptions, index + variant, 2)),
    deliveryTags: unique(pickFrom(deliveryOptions, index + variant, 2)),
    serviceTags: unique(pickFrom(serviceOptions, index + variant, 2)),
    image: blueprint.image,
    accent: blueprint.accent,
  };
}

const targetMerchantCount = 150;
const generatedMerchants = Array.from({ length: targetMerchantCount - merchants.length - extraMerchants.length }, (_, index) =>
  generatedMerchant(index),
);
uniquifyGeneratedNames(generatedMerchants);

merchants.push(...extraMerchants, ...generatedMerchants);

const historyOrders = [
  { merchantId: "m2", daysAgo: 3, weight: 0.88 },
  { merchantId: "m5", daysAgo: 5, weight: 0.74 },
  { merchantId: "m10", daysAgo: 9, weight: 0.52 },
  { merchantId: "m7", daysAgo: 12, weight: 0.39 },
  { merchantId: "m23", daysAgo: 15, weight: 0.34 },
  { merchantId: "m34", daysAgo: 18, weight: 0.28 },
  { merchantId: "m37", daysAgo: 23, weight: 0.21 },
];

const relationLabels = {
  category: "品类",
  cuisine: "菜系",
  tastes: "口味",
  scenes: "场景",
  area: "商圈",
  priceBucket: "价格",
  foods: "菜品",
  ingredients: "食材",
  cooking: "做法",
  nutrition: "营养",
  timeSlots: "时段",
  crowd: "人群",
  portions: "份量",
  benefits: "优惠",
  deliveryTags: "配送",
  serviceTags: "服务",
};

const relationConfig = {
  category: { label: "品类", weight: 1.24 },
  cuisine: { label: "菜系", weight: 1.08 },
  tastes: { label: "口味", weight: 1.32 },
  scenes: { label: "场景", weight: 1.06 },
  area: { label: "商圈", weight: 0.54 },
  priceBucket: { label: "价格", weight: 0.7 },
  foods: { label: "菜品", weight: 1.42 },
  ingredients: { label: "食材", weight: 1.18 },
  cooking: { label: "做法", weight: 0.92 },
  nutrition: { label: "营养", weight: 0.86 },
  timeSlots: { label: "时段", weight: 0.76 },
  crowd: { label: "人群", weight: 0.9 },
  portions: { label: "份量", weight: 0.72 },
  benefits: { label: "优惠", weight: 0.5 },
  deliveryTags: { label: "配送", weight: 0.45 },
  serviceTags: { label: "服务", weight: 0.48 },
};

const cuisineDefaults = {
  川菜: { ingredients: ["牛肉", "豆皮", "辣椒"], cooking: ["现煮", "红油"], nutrition: ["高蛋白"], crowd: ["重口味人群"] },
  粤式: { ingredients: ["鸡肉", "米饭", "青菜"], cooking: ["烧腊", "蒸点"], nutrition: ["高蛋白"], crowd: ["工作餐人群"] },
  健康餐: { ingredients: ["鸡胸肉", "生菜", "藜麦"], cooking: ["少油", "冷食"], nutrition: ["低脂", "高蛋白"], crowd: ["健身人群", "轻食人群"] },
  西式快餐: { ingredients: ["鸡肉", "芝士", "土豆"], cooking: ["炸制", "烘烤"], nutrition: ["高热量"], crowd: ["夜宵人群"] },
  潮汕: { ingredients: ["牛肉", "米粉", "鲜虾"], cooking: ["现煮", "砂锅"], nutrition: ["暖胃", "高蛋白"], crowd: ["汤粉偏好"] },
  东北菜: { ingredients: ["猪肉", "土豆", "青椒"], cooking: ["现炒", "炖煮"], nutrition: ["大份量"], crowd: ["合餐人群"] },
  广式甜品: { ingredients: ["芒果", "牛奶", "椰汁"], cooking: ["冷饮", "甜品"], nutrition: ["高糖"], crowd: ["甜品偏好"] },
  西北: { ingredients: ["面粉", "牛肉", "辣椒"], cooking: ["手工面", "凉拌"], nutrition: ["暖胃"], crowd: ["面食偏好"] },
  云南米线: { ingredients: ["米线", "牛肉", "番茄"], cooking: ["现煮", "汤粉"], nutrition: ["暖胃"], crowd: ["粉面偏好"] },
  粥点: { ingredients: ["大米", "虾仁", "瘦肉"], cooking: ["慢煮", "蒸点"], nutrition: ["暖胃"], crowd: ["清淡人群"] },
  韩式: { ingredients: ["米饭", "泡菜", "牛肉"], cooking: ["拌饭", "炒制"], nutrition: ["高蛋白"], crowd: ["异国料理偏好"] },
  江浙: { ingredients: ["猪肉", "面粉", "虾仁"], cooking: ["蒸点", "煎制"], nutrition: ["小份多样"], crowd: ["点心偏好"] },
  海南菜: { ingredients: ["鸡肉", "米饭", "冬瓜"], cooking: ["白切", "盖饭"], nutrition: ["高蛋白"], crowd: ["工作餐人群"] },
  意式: { ingredients: ["芝士", "番茄", "面粉"], cooking: ["烤制", "烘烤"], nutrition: ["高热量"], crowd: ["合餐人群"] },
  健康甜品: { ingredients: ["酸奶", "水果", "燕麦"], cooking: ["冷食", "现切"], nutrition: ["低脂"], crowd: ["甜品偏好", "健身人群"] },
  贵州菜: { ingredients: ["鱼肉", "酸汤", "折耳根"], cooking: ["炖煮", "酸汤"], nutrition: ["高蛋白"], crowd: ["鱼鲜偏好"] },
};

const categoryDefaults = {
  川湘辣味: { timeSlots: ["午高峰", "晚高峰"], portions: ["单人套餐", "双人餐"], benefits: ["满减"], deliveryTags: ["保温包装"], serviceTags: ["辣度可选"] },
  盖饭便当: { timeSlots: ["午高峰", "晚高峰"], portions: ["单人套餐", "加量饭"], benefits: ["满减"], deliveryTags: ["出餐快"], serviceTags: ["米饭可加"] },
  轻食沙拉: { timeSlots: ["午高峰", "下午"], portions: ["单人套餐"], benefits: ["新客券"], deliveryTags: ["冷链包装"], serviceTags: ["酱料分装"] },
  炸鸡汉堡: { timeSlots: ["晚高峰", "夜间"], portions: ["双人餐", "多人分享"], benefits: ["套餐优惠"], deliveryTags: ["夜间配送"], serviceTags: ["保温包装"] },
  粉面汤粥: { timeSlots: ["早餐", "午高峰", "晚高峰"], portions: ["单人套餐"], benefits: ["折扣套餐"], deliveryTags: ["汤面分装"], serviceTags: ["汤底分装"] },
  家常小炒: { timeSlots: ["午高峰", "晚高峰"], portions: ["双人餐", "多人分享"], benefits: ["满减"], deliveryTags: ["保温包装"], serviceTags: ["米饭可加"] },
  甜品饮品: { timeSlots: ["下午", "夜间"], portions: ["单杯", "双杯套餐"], benefits: ["第二杯半价"], deliveryTags: ["近距离"], serviceTags: ["糖度可选"] },
  地方小吃: { timeSlots: ["早餐", "午高峰", "夜间"], portions: ["单人套餐", "小份尝鲜"], benefits: ["免配送费"], deliveryTags: ["出餐快"], serviceTags: ["酱料分装"] },
  早餐粥点: { timeSlots: ["早餐", "夜间"], portions: ["单人套餐", "双人餐"], benefits: ["早餐套餐"], deliveryTags: ["保温包装"], serviceTags: ["粥料分装"] },
  日韩料理: { timeSlots: ["午高峰", "下午"], portions: ["单人套餐", "小份尝鲜"], benefits: ["会员券"], deliveryTags: ["冷链包装"], serviceTags: ["配料分装"] },
  包子点心: { timeSlots: ["早餐", "午高峰"], portions: ["单人套餐", "家庭份"], benefits: ["折扣套餐"], deliveryTags: ["保温包装"], serviceTags: ["蘸料分装"] },
  披萨意面: { timeSlots: ["晚高峰", "夜间"], portions: ["双人餐", "家庭份"], benefits: ["套餐优惠"], deliveryTags: ["保温包装"], serviceTags: ["切片配送"] },
  水果酸奶: { timeSlots: ["早餐", "下午"], portions: ["单杯", "小份尝鲜"], benefits: ["会员券"], deliveryTags: ["冷链包装"], serviceTags: ["糖度可选"] },
  特色正餐: { timeSlots: ["晚高峰"], portions: ["双人餐", "多人分享"], benefits: ["满减"], deliveryTags: ["保温包装"], serviceTags: ["汤汁防漏"] },
};

const tasteAliases = {
  麻辣: ["香辣", "重口味", "辣度可选"],
  香辣: ["麻辣", "重口味"],
  酸辣: ["香辣", "重口味", "开胃"],
  清淡: ["清爽", "暖胃", "低脂"],
  清爽: ["清淡", "低脂"],
  甜口: ["奶香", "甜品偏好"],
  咸鲜: ["鲜香", "下饭"],
  鲜香: ["咸鲜", "汤面"],
  芝士: ["奶香", "高热量"],
  肉食: ["高蛋白", "下饭"],
  面食: ["粉面偏好", "暖胃"],
  汤面: ["汤粉", "暖胃"],
  低脂: ["轻食", "控糖", "健身人群"],
  高热量: ["多人分享", "夜宵人群"],
};

function unique(values) {
  return [...new Set((values || []).filter(Boolean))];
}

function normalizeMerchant(merchant) {
  const byCuisine = cuisineDefaults[merchant.cuisine] || {};
  const byCategory = categoryDefaults[merchant.category] || {};
  merchant.ingredients = unique([...(merchant.ingredients || []), ...(byCuisine.ingredients || [])]);
  merchant.cooking = unique([...(merchant.cooking || []), ...(byCuisine.cooking || [])]);
  merchant.nutrition = unique([...(merchant.nutrition || []), ...(byCuisine.nutrition || [])]);
  merchant.timeSlots = unique([...(merchant.timeSlots || []), ...(byCategory.timeSlots || [])]);
  merchant.crowd = unique([...(merchant.crowd || []), ...(byCuisine.crowd || [])]);
  merchant.portions = unique([...(merchant.portions || []), ...(byCategory.portions || [])]);
  merchant.benefits = unique([...(merchant.benefits || []), ...(byCategory.benefits || [])]);
  merchant.deliveryTags = unique([...(merchant.deliveryTags || []), ...(byCategory.deliveryTags || [])]);
  merchant.serviceTags = unique([...(merchant.serviceTags || []), ...(byCategory.serviceTags || [])]);
  merchant.timeSlots = unique([...merchant.timeSlots, ...merchant.scenes.filter((scene) => ["早餐", "下午茶", "夜宵"].includes(scene))]);
  merchant.crowd = unique([...merchant.crowd, ...merchant.scenes.filter((scene) => ["工作餐", "多人餐", "健身", "病中餐"].includes(scene)).map((scene) => `${scene}人群`)]);
  merchant.tastes = unique(merchant.tastes);
  merchant.foods = unique(merchant.foods);
  return merchant;
}

merchants.forEach(normalizeMerchant);

let featureDocumentFrequency = new Map();

const state = {
  clicks: [],
  round: 0,
  seed: 17,
  recommendations: [],
};

const nodes = {
  list: document.querySelector("#merchantList"),
  graph: document.querySelector("#graphSvg"),
  round: document.querySelector("#roundText"),
  state: document.querySelector("#stateText"),
  temp: document.querySelector("#tempStrength"),
  coverage: document.querySelector("#coverageText"),
  taste: document.querySelector("#topTaste"),
  entity: document.querySelector("#entityText"),
  relation: document.querySelector("#relationText"),
  edge: document.querySelector("#edgeText"),
  dataset: document.querySelector("#datasetSizeText"),
  trace: document.querySelector("#traceCards"),
  reset: document.querySelector("#resetBtn"),
};

function priceBucket(price) {
  if (price < 25) return "低价";
  if (price < 36) return "平价";
  if (price < 48) return "中价";
  return "聚餐价";
}

function merchantFeatures(merchant) {
  const fields = {
    category: [merchant.category],
    cuisine: [merchant.cuisine],
    area: [merchant.area],
    priceBucket: [priceBucket(merchant.price)],
    tastes: merchant.tastes,
    scenes: merchant.scenes,
    foods: merchant.foods,
    ingredients: merchant.ingredients,
    cooking: merchant.cooking,
    nutrition: merchant.nutrition,
    timeSlots: merchant.timeSlots,
    crowd: merchant.crowd,
    portions: merchant.portions,
    benefits: merchant.benefits,
    deliveryTags: merchant.deliveryTags,
    serviceTags: merchant.serviceTags,
  };
  return Object.entries(fields).flatMap(([type, values]) =>
    unique(values).map((value) => ({
      type,
      value,
      relWeight: relationConfig[type]?.weight || 1,
    })),
  );
}

function featureKey(feature) {
  return `${feature.type}:${feature.value}`;
}

function buildFeatureDocumentFrequency() {
  const counts = new Map();
  for (const merchant of merchants) {
    const seen = new Set(merchantFeatures(merchant).map(featureKey));
    for (const key of seen) {
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  featureDocumentFrequency = counts;
}

function idfFor(feature) {
  const count = featureDocumentFrequency.get(featureKey(feature)) || 1;
  return 1 + Math.log((merchants.length + 1) / (count + 1));
}

function relatedFeatures(feature) {
  const related = tasteAliases[feature.value] || [];
  if (!related.length) return [];
  return related.map((value) => ({
    type: ["低脂", "高热量", "暖胃", "高蛋白", "控糖", "轻食"].includes(value)
      ? "nutrition"
      : ["健身人群", "夜宵人群", "甜品偏好", "粉面偏好"].includes(value)
        ? "crowd"
        : feature.type,
    value,
    relWeight: (relationConfig[feature.type]?.weight || 1) * 0.68,
    inferred: true,
  }));
}

buildFeatureDocumentFrequency();

function buildInterestMap() {
  const map = new Map();
  const add = (feature, amount, source, label, inferred = false) => {
    const key = featureKey(feature);
    const previous = map.get(key) || {
      ...feature,
      click: 0,
      history: 0,
      inferredClick: 0,
      inferredHistory: 0,
      total: 0,
      labels: new Set(),
    };
    previous[source] += amount;
    if (inferred && source === "click") previous.inferredClick += amount;
    if (inferred && source === "history") previous.inferredHistory += amount;
    previous.total += amount * (feature.relWeight || relationConfig[feature.type]?.weight || 1);
    previous.labels.add(label);
    map.set(key, previous);
  };

  if (state.round > 0) {
    for (const order of historyOrders) {
      const merchant = merchants.find((item) => item.id === order.merchantId);
      const decay = Math.exp(-order.daysAgo / 18) * order.weight;
      for (const feature of merchantFeatures(merchant)) {
        const amount = decay * feature.relWeight * Math.min(2.1, idfFor(feature));
        add(feature, amount, "history", `${merchant.name} ${order.daysAgo}天前`);
        for (const related of relatedFeatures(feature)) {
          add(related, amount * 0.28, "history", `${merchant.name} ${order.daysAgo}天前`, true);
        }
      }
    }
  }

  state.clicks.forEach((click, index) => {
    const merchant = merchants.find((item) => item.id === click.merchantId);
    const recency = Math.pow(0.72, state.clicks.length - 1 - index);
    for (const feature of merchantFeatures(merchant)) {
      const amount = recency * click.weight * feature.relWeight * Math.min(2.2, idfFor(feature));
      add(feature, amount, "click", `${merchant.name} 刚刚点击`);
      for (const related of relatedFeatures(feature)) {
        add(related, amount * 0.34, "click", `${merchant.name} 刚刚点击`, true);
      }
    }
  });

  return map;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function stableNoise(id) {
  let hash = state.seed;
  for (let index = 0; index < id.length; index += 1) {
    hash = (hash * 31 + id.charCodeAt(index)) % 997;
  }
  return (hash % 100) / 1000;
}

function scoreMerchant(merchant, interests) {
  const features = merchantFeatures(merchant);
  let clickScore = 0;
  let histScore = 0;
  let overlapScore = 0;
  const matched = [];
  for (const feature of features) {
    const interest = interests.get(featureKey(feature));
    if (!interest) continue;
    const rarity = Math.sqrt(Math.min(2.2, idfFor(feature)));
    const contribution = (1.18 * interest.click + 0.82 * interest.history) * rarity;
    clickScore += interest.click * rarity;
    histScore += interest.history * rarity;
    overlapScore += contribution;
    matched.push({
      ...feature,
      click: interest.click,
      history: interest.history,
      inferredClick: interest.inferredClick,
      inferredHistory: interest.inferredHistory,
      idf: rarity,
      total: contribution,
    });
  }

  const normalizedClick = 1 - Math.exp(-clickScore / 42);
  const normalizedHist = 1 - Math.exp(-histScore / 54);
  const normalizedOverlap = 1 - Math.exp(-overlapScore / 72);
  const quality = merchant.rating / 5;
  const popularity = Math.log1p(merchant.monthly) / Math.log1p(3200);
  const distanceFit = Math.max(0, 1 - merchant.distance / 3.5);
  const deliveryFit = Math.max(0, 1 - merchant.delivery / 8);
  const relationCoverage = new Set(matched.map((item) => item.type)).size / Object.keys(relationConfig).length;
  const freshnessBoost = state.clicks.some((click) => click.merchantId === merchant.id) ? -0.025 : 0.018 * (1 - relationCoverage);
  const total =
    state.round === 0
      ? stableNoise(merchant.id) + Math.random() * 0.4
      : Math.max(
          0,
          Math.min(
            0.98,
            0.34 * normalizedClick +
              0.22 * normalizedHist +
              0.18 * normalizedOverlap +
              0.08 * relationCoverage +
              0.07 * quality +
              0.05 * popularity +
              0.035 * distanceFit +
              0.02 * deliveryFit +
              stableNoise(merchant.id) * 0.35 +
              freshnessBoost,
          ),
        );

  return {
    merchant,
    total,
    click: normalizedClick,
    history: normalizedHist,
    overlap: normalizedOverlap,
    coverage: relationCoverage,
    base: Math.min(1, 0.45 * quality + 0.35 * popularity + 0.2 * distanceFit),
    matched: matched.sort((a, b) => b.total - a.total).slice(0, 8),
  };
}

function rerank() {
  const interests = buildInterestMap();
  state.recommendations = merchants
    .map((merchant) => scoreMerchant(merchant, interests))
    .sort((a, b) => b.total - a.total);
}

function reasonText(scored) {
  if (state.round === 0) {
    return ["随机探索", `${scored.merchant.area}`, `${scored.merchant.category}`];
  }
  const reasons = scored.matched.slice(0, 3).map((item) => `${relationLabels[item.type]}:${item.value}`);
  if (reasons.length < 3) reasons.push(`评分 ${scored.merchant.rating}`);
  if (reasons.length < 3) reasons.push(`月售 ${scored.merchant.monthly}`);
  return reasons.slice(0, 3);
}

function renderMerchants() {
  nodes.list.innerHTML = "";
  state.recommendations.forEach((scored, index) => {
    const merchant = scored.merchant;
    const card = document.createElement("button");
    card.className = `merchant-card ${state.clicks.some((click) => click.merchantId === merchant.id) ? "clicked" : ""}`;
    card.type = "button";
    card.addEventListener("click", () => selectMerchant(merchant.id));
    card.innerHTML = `
      <span class="rank-badge">${index + 1}</span>
      <img class="thumb" alt="${merchant.category}" src="${merchant.image}" />
      <div class="merchant-body">
        <div class="merchant-title-row">
          <h3 class="merchant-title"><span>${merchant.name}</span></h3>
          <span class="score-chip">${(scored.total * 100).toFixed(0)}</span>
        </div>
        <div class="merchant-meta">
          <span class="tag">${merchant.category}</span>
          <span class="tag">${merchant.cuisine}</span>
          <span class="tag hot">月售 ${merchant.monthly}</span>
          <span class="tag price">¥${merchant.price} 起</span>
          <span class="tag">${merchant.area} ${merchant.distance}km</span>
        </div>
        <div class="reasons">
          ${reasonText(scored)
            .map((reason) => `<span class="reason-pill">${reason}</span>`)
            .join("")}
        </div>
        <div class="score-bars">
          ${bar("点击", scored.click, "click")}
          ${bar("历史", scored.history, "hist")}
          ${bar("图谱", scored.overlap, "kg")}
          ${bar("基础", scored.base, "base")}
        </div>
      </div>
    `;
    nodes.list.appendChild(card);
  });
}

function bar(label, value, className) {
  return `
    <div>
      <span class="bar-label"><span>${label}</span><span>${value.toFixed(2)}</span></span>
      <span class="bar-track"><i class="bar-fill ${className}" style="width:${Math.max(5, value * 100)}%"></i></span>
    </div>
  `;
}

function selectMerchant(id) {
  state.clicks.push({
    merchantId: id,
    time: Date.now(),
    weight: 1 + Math.min(0.4, state.clicks.length * 0.08),
  });
  state.round += 1;
  rerank();
  render();
}

function reset() {
  state.clicks = [];
  state.round = 0;
  state.seed = (state.seed + 13) % 101;
  rerank();
  render();
}

function topInterests(interests, count = 10) {
  return Array.from(interests.values())
    .filter((item) => item.total > 0.05)
    .sort((a, b) => b.total - a.total)
    .slice(0, count);
}

function renderGraph() {
  const interests = buildInterestMap();
  const top = topInterests(interests, state.round ? 18 : 0);
  nodes.graph.innerHTML = "";

  const svg = nodes.graph;
  const make = (name, attrs = {}) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    return el;
  };

  const defs = make("defs");
  defs.innerHTML = `
    <linearGradient id="userGrad" x1="0" x2="1">
      <stop offset="0%" stop-color="#3fb984" />
      <stop offset="100%" stop-color="#519be8" />
    </linearGradient>
    <linearGradient id="clickGrad" x1="0" x2="1">
      <stop offset="0%" stop-color="#f56a4d" />
      <stop offset="100%" stop-color="#f4b63f" />
    </linearGradient>
  `;
  svg.appendChild(defs);

  const user = { x: 450, y: state.round ? 275 : 310 };
  const historyMerchants = (state.round ? historyOrders.slice(0, 5) : []).map((order, index) => {
    const merchant = merchants.find((item) => item.id === order.merchantId);
    return {
      id: merchant.id,
      label: merchant.name.replace(/(.{5}).+/, "$1…"),
      sub: `${order.daysAgo}天前`,
      x: 115 + index * 118,
      y: 95 + (index % 2) * 54,
      color: "#519be8",
      weight: order.weight,
    };
  });
  const clickMerchants = state.clicks.slice(-4).map((click, index, arr) => {
    const merchant = merchants.find((item) => item.id === click.merchantId);
    return {
      id: merchant.id,
      label: merchant.name.replace(/(.{5}).+/, "$1…"),
      sub: "点击",
      x: 625 + index * (210 / Math.max(1, arr.length - 1 || 1)),
      y: 95 + (index % 2) * 62,
      color: merchant.accent,
      weight: click.weight,
    };
  });

  for (const node of historyMerchants) {
    drawCurve(user, node, "history", 2 + node.weight * 3, "pref_hist");
  }
  for (const node of clickMerchants) {
    drawCurve(user, node, "click", 3 + node.weight * 4, "pref_click");
  }

  if (state.round === 0) {
    svg.appendChild(make("text", { x: 450, y: 420, class: "empty-hint" })).textContent = "初始图谱仅包含用户节点";
    svg.appendChild(make("text", { x: 450, y: 452, class: "empty-sub" })).textContent =
      "点击任意商家后，临时兴趣节点会进入图谱并参与重排";
  }

  const featureNodes = top.map((interest, index) => {
    const col = index % 6;
    const row = Math.floor(index / 6);
    return {
      ...interest,
      x: 90 + col * 145 + (row % 2) * 34,
      y: 385 + row * 72,
      color: colorForFeature(interest),
    };
  });

  for (const feature of featureNodes) {
    const width = 1.5 + Math.min(6, feature.total * 1.6);
    drawCurve(user, feature, feature.click > feature.history ? "click" : "attr", width, relationLabels[feature.type]);
  }

  for (const node of historyMerchants) drawNode(node, 30, "#fff");
  for (const node of clickMerchants) drawNode(node, 34, "#fff");
  drawNode({ ...user, label: "用户 U_demo", sub: "当前会话", color: "url(#userGrad)" }, 46, "#fff");
  for (const feature of featureNodes) {
    drawNode(
      {
        ...feature,
        label: feature.value,
        sub: `${relationLabels[feature.type]} ${(feature.total).toFixed(1)}`,
      },
      28 + Math.min(10, feature.total * 1.5),
      "#fff",
    );
  }

  function drawCurve(from, to, className, width, label) {
    const midY = (from.y + to.y) / 2;
    const path = make("path", {
      d: `M ${from.x} ${from.y} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y}`,
      class: `graph-link ${className}`,
      "stroke-width": width,
      opacity: 0.9,
    });
    svg.appendChild(path);
    if (state.round || className !== "attr") {
      const lx = (from.x + to.x) / 2;
      const ly = (from.y + to.y) / 2 - 8;
      const text = make("text", { x: lx, y: ly, class: "edge-label" });
      text.textContent = label;
      svg.appendChild(text);
    }
  }

  function drawNode(node, radius) {
    const group = make("g");
    const circle = make("circle", {
      cx: node.x,
      cy: node.y,
      r: radius,
      class: "node-circle",
      fill: node.color,
    });
    const label = make("text", { x: node.x, y: node.y + radius + 21, class: "node-label" });
    label.textContent = node.label;
    const sub = make("text", { x: node.x, y: node.y + radius + 38, class: "node-sub" });
    sub.textContent = node.sub;
    group.append(circle, label, sub);
    svg.appendChild(group);
  }
}

function colorForFeature(item) {
  const colors = {
    category: "#3fb984",
    cuisine: "#519be8",
    tastes: "#f56a4d",
    scenes: "#f4b63f",
    area: "#8f74d4",
    priceBucket: "#c667a6",
    foods: "#2f9f8f",
    ingredients: "#df7c42",
    cooking: "#6a9f50",
    nutrition: "#5d85d8",
    timeSlots: "#d49d2f",
    crowd: "#b46dc3",
    portions: "#8b8f4d",
    benefits: "#d75f83",
    deliveryTags: "#4f9fc6",
    serviceTags: "#7a83cc",
  };
  return colors[item.type] || "#6f7b8c";
}

function renderTrace() {
  nodes.trace.innerHTML = "";
  const cards = state.recommendations.slice(0, 4);
  if (state.round === 0) {
    nodes.trace.innerHTML = `
      <article class="trace-card">
        <div class="trace-score"><h3>等待交互</h3><span>cold start</span></div>
        <p>初始推荐采用随机探索和商家基础质量。点击一个商家后，系统会把它的品类、口味、场景、商圈等写入临时兴趣图谱。</p>
      </article>
    `;
    return;
  }
  cards.forEach((scored, index) => {
    const card = document.createElement("article");
    card.className = "trace-card";
    const topReasons = scored.matched.slice(0, 4);
    const path =
      topReasons.length > 0
        ? topReasons
            .map(
              (item) =>
                `用户 → ${item.click >= item.history ? "点击兴趣" : "历史兴趣"} → ${relationLabels[item.type]}:${item.value}`,
            )
            .join("；")
        : "与当前兴趣无直接重合，主要由基础质量和探索项进入候选。";
    card.innerHTML = `
      <div class="trace-score">
        <h3>#${index + 1} ${scored.merchant.name}</h3>
        <span>${(scored.total * 100).toFixed(1)}</span>
      </div>
      <p>${path}</p>
    `;
    nodes.trace.appendChild(card);
  });
}

function graphStats(interests) {
  const entities = new Set();
  let merchantAttributeEdges = 0;
  for (const merchant of merchants) {
    entities.add(`merchant:${merchant.id}`);
    for (const feature of merchantFeatures(merchant)) {
      entities.add(featureKey(feature));
      merchantAttributeEdges += 1;
    }
  }
  entities.add("user:demo");
  const activeInterestEdges =
    state.round === 0 ? 0 : historyOrders.length + state.clicks.length + Array.from(interests.values()).filter((item) => item.total > 0.05).length;
  return {
    entities: entities.size,
    relations: Object.keys(relationConfig).length,
    edges: merchantAttributeEdges + activeInterestEdges,
  };
}

function renderMetrics() {
  const interests = buildInterestMap();
  const current = topInterests(interests, 18);
  const clickStrength = current.reduce((sum, item) => sum + item.click, 0);
  const top = current[0];
  const stats = graphStats(interests);
  nodes.dataset.textContent = `子集样本 ${merchants.length} 家`;
  nodes.round.textContent = `第 ${state.round} 轮`;
  nodes.state.textContent =
    state.round === 0
      ? "初始随机推荐，等待首次点击"
      : `已点击 ${state.clicks.length} 次，${merchants.length} 家商家按动态图谱兴趣整体重排`;
  nodes.temp.textContent = clickStrength.toFixed(2);
  nodes.coverage.textContent = `${new Set(current.map((item) => relationLabels[item.type])).size} 类`;
  nodes.taste.textContent = top ? top.value : "等待点击";
  nodes.entity.textContent = `${stats.entities} 个`;
  nodes.relation.textContent = `${stats.relations} 类`;
  nodes.edge.textContent = `${stats.edges} 条`;
}

function render() {
  renderMetrics();
  renderMerchants();
  renderGraph();
  renderTrace();
}

nodes.reset.addEventListener("click", reset);
rerank();
render();
