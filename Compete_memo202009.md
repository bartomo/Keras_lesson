教師あり回帰予測問題

１．問題定義
79項目の説明変数
価格が目的変数

２．データセット作成
データセット作成
前処理
ワンホットエンコーディング
欠損処理

２＿１成功指標の選択
二乗平均平方根誤差（RMSE)

２＿２評価プロトコル
検証方法
・k分割交差検証

３．特徴量エンジニアリング

４．ベースラインを超える性能モデルを開発

５．過学習するモデルの開発



#ラベル日本語訳
SalePrice-プロパティのドルでの販売価格。これは、予測しようとしているターゲット変数です。
Id:
MSSubClass：建物クラス
MSZoning：一般的なゾーニング分類
LotFrontage：プロパティに接続された道路の線形フィート
LotArea：ロットサイズ（平方フィート）
Street：道路アクセスのタイプ
Alley:路地：路地アクセスのタイプ
LotShape：プロパティの一般的な形状
LandContour：プロパティの平坦性
Utilities:ユーティリティ：利用可能なユーティリティのタイプ
LotConfig：ロット構成
LandSlope：プロパティの勾配
Neighborhood:近所：エイムス市内の制限内の物理的な場所
Condition1:条件1：幹線道路または鉄道に近接
Condition2:条件2：幹線道路または鉄道に近接している（秒がある場合）
BldgType：住居のタイプ
HouseStyle：住居のスタイル
OverallQual：全体的な材料と仕上げの品質
OverallCond：全体的な状態の評価
YearBuilt：元の建設日
YearRemodAdd：改造日
RoofStyle：屋根のタイプ
RoofMatl：屋根のマテリアル
Exterior1st：家の外装
Exterior2nd：家の外装（複数の材料の場合）
MasVnrType：石積みのベニヤのタイプ
MasVnrArea：平方フィートの石材ベニヤ面積
ExterQual：外装材の品質
ExterCond：外装材の現状
Foundation:財団：財団の種類
BsmtQual：地下室の高さ
BsmtCond：地下室の概況
BsmtExposure：ストライキまたは庭園レベルの地下壁
BsmtFinType1：地下の仕上がり面積
BsmtFinSF1：タイプ1仕上げ済み平方フィート
BsmtFinType2：2番目に終了した領域の品質（存在する場合）
BsmtFinSF2：タイプ2仕上げ済み平方フィート
BsmtUnfSF：地下室の未完成の平方フィート
TotalBsmtSF：地下面積の合計平方フィート
Heating:暖房：暖房の種類
HeatingQC：加熱の品質と状態
CentralAir：セントラル空調
Electrical:電気：電気システム
1stFlrSF：1階の平方フィート
2ndFlrSF：2階の平方フィート
LowQualFinSF：低品質の仕上げ済み平方フィート（すべてのフロア）
GrLivArea：グレード（地上）のリビングエリアの平方フィート
BsmtFullBath：地下フルバスルーム
BsmtHalfBath：地下半分のバスルーム
FullBath：グレードを超えるフルバスルーム
HalfBath：グレード上記ハーフ浴場
BedroomAbvGr:ベッドルーム：地下階の上のベッドルームの数
KitchenAbvGr:キッチン：キッチン数
KitchenQual：キッチンの品質
TotRmsAbvGrd：グレードを超える部屋の合計（バスルームは含まれません）
Functional:機能的：ホーム機能の評価
Fireplace:暖炉：暖炉の数
FireplaceQu：暖炉の品質
GarageType：ガレージの場所
GarageYrBlt：ガレージが建設された年
GarageFinish：ガレージの内部仕上げ
GarageCars：車の容量でのガレージのサイズ
GarageArea：ガレージの平方フィートでのサイズ
GarageQual：ガレージの品質
GarageCond：ガレージの状態
PavedDrive：舗装された私道
WoodDeckSF：平方フィートのウッドデッキ領域
OpenPorchSF：平方フィートのオープンポーチエリア
EnclosedPorch：平方フィートで囲まれたポーチエリア
3SsnPorch：平方フィートのスリーシーズンポーチエリア
ScreenPorch：平方フィートのスクリーンポーチエリア
PoolArea：平方フィートのプール面積
PoolQC：プールの品質
Fence:フェンス：フェンスの品質
MiscFeature：他のカテゴリでカバーされていないその他の機能
MiscVal：その他の機能の$ Value
MoSold：販売月
YrSold：販売年
SaleType：販売のタイプ
SaleCondition：販売の条件
データセットのページに特徴量の日本語訳がのっていたので掲載しておきます。