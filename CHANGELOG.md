# Changelog

## [0.3.0](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/compare/v0.2.3...v0.3.0) (2024-10-25)


### Features

* Remove support for Python 3.8 ([#97](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/97)) ([ba02bf5](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/ba02bf528830dafb599afae43bd1bf3b2ac7c493))


### Documentation

* Update README.rst ([#94](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/94)) ([fbc3e5c](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/fbc3e5c9f65f49a9a9f6a505e5992c8c2a80fe61))

## [0.2.3](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/compare/v0.2.2...v0.2.3) (2024-07-03)


### Bug Fixes

* Use lazy refresh for Cloud SQL Connector ([#86](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/86)) ([58591d5](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/58591d564255217f00582eabd12a1183b089f5e3))

## [0.2.2](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/compare/v0.2.1...v0.2.2) (2024-04-30)


### Documentation

* Add API reference docs ([#71](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/71)) ([7b39d29](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/7b39d29f0e20e47f15d2167af73fce71f4fb9e18))
* Add example workflow to vector store doc ([#67](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/67)) ([fadfc41](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/fadfc41d0f3c907262f2c3936205cf68204c3909))

## [0.2.1](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/compare/v0.2.0...v0.2.1) (2024-04-15)


### Bug Fixes

* Allow similarity search on table names with special characters ([#65](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/65)) ([4832524](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/4832524b4b5ab9eddeb2a1bdc919608f59945652))

## [0.2.0](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/compare/v0.1.0...v0.2.0) (2024-04-08)


### Features

* Add index types for vector search ([#55](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/55)) ([2e30b48](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/2e30b48ad2d1fb11f5f8964808ed5143d9231084))
* Add MySQLVectorStore initialization methods ([#52](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/52)) ([a1c9411](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/a1c941149e1f1b33991b997e5236c4a7971058fd))
* Adding search functions and tests ([#56](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/56)) ([5b80694](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/5b806947e5c827ebca553a68ff74a14c7d22a6a5))
* **ci:** Run tests against multiple versions ([#51](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/51)) ([3439c9d](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/3439c9d6a277a95da835f1c59d4727855a187dee))
* Support add and delete from MySQLVectorStore ([#53](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/53)) ([ce45617](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/ce45617ae6c9f1b6e539c31e4bcdd47aa7daf964))


### Documentation

* Add basic MySQLVectorStore usage to README ([#58](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/58)) ([e871c2b](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/e871c2b503fb0d056d7e374394db36e44dcda4c2))
* Add end-to-end MySQL quickstart ([#61](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/61)) ([388f5a4](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/388f5a4e6b76d23c1e683029c5ea034cfe84bbf7))
* Add github links ([#46](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/46)) ([54fbab5](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/54fbab5fd41e7b49a2d5da800afad5d3fb66b40c))
* Add MySQLVectorStore reference notebook ([#59](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/59)) ([0ece837](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/0ece837e98ff60512d26b5c7c8fb4803e056ad3c))

## 0.1.0 (2024-02-22)


### Features

* Add `MySQLChatMessageHistory` class ([#13](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/13)) ([b107a43](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/b107a430f0f257d2e91d3c47933b395c63ce7d6b))
* Add document saver class and support save/load/delete user journey ([#16](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/16)) ([49f6018](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/49f6018f92a140340fae9139f22e7c6244c22fac))
* Add MySQLEngine and Loader load functionality ([#9](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/9)) ([6c8af85](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/6c8af85a2676ca06e41edfdd67cc497eca9b7107))


### Documentation

* Add chat message history docs ([#21](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/21)) ([5aecbd0](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/5aecbd0af8707f0669048e3a1ac1f388a1290bc7))
* Add docloader codelab ([#18](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/18)) ([6c82c4e](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/6c82c4e02ba4a4d344848c8d45b1bc19d7c19080))
* Add MySQLChatMessageHistory to readme ([#33](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/issues/33)) ([17c5714](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/commit/17c571433cea7f740ae249d4ceb7c35b308fb112))
