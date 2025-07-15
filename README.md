ポートフォリオ用に公開しました、大学の学士論文のために作ったCLIプログラムです。

このプログラムはクリーンコードをサポートするためにAIを使ってJavaのソースコードで関数名を入力した場合、戻り値や引数をユーザーに提案します。Visual Studio Code(以降VSC)などの拡張機能として作られています。プロジェクト自体は以下の三つのソフトで構成されています。

- [Java Crawler](https://github.com/jkahlem/javacrawler): Javaのソースコードを解析して、必要なデータや構成要素を抽出しXMLに変換するプログラムです
- [Cleancode Util](https://github.com/jkahlem/cleancodeutil): Goで書かれたソフトで、以下の二つのソフトに分けられています
  - Data Extractor: トレーニングセットや検証セットを作るCLIです。指定されたレポジトリをクローンして、Java Crawlerを使ってデータを抽出・解析し、決まったルールでデータセットを作成します。それらのセットをPredictorに送信することでAIを学習させられます。
  - Language Server: Language Server Protocol (LSP)を実装したサーバーアプリで、リアルタイムでプロジェクトの解析をし、入力される関数名をPredictorに送信し、推測された戻り値や引数を提案としてVSCに送ります。
- [Predictor](https://github.com/jkahlem/predictor): **(このプロジェクト)** AIモデルの操作（学習・推測）のためのプログラムです。 

このプログラムは[Cleancode Util](https://github.com/jkahlem/cleancodeutil)のプログラムからJsonRPCのメッセージを受けて動作するように作られているので、単体では機能しません。エントリーポイントはこちらのファイルです: https://github.com/jkahlem/predictor/blob/master/predictor.py#L200
