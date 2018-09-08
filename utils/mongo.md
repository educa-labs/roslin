# Example

```python
model = KNNPredictor()  # Example of model to be saved...
model.save('test-database', 'knn')
```
# MongoDB on Terminal

- **Start MongoDB**

```bash
sudo service mongod start
```

- **Verify that MongoDB has started successfully**

```bash
cat /var/log/mongodb/mongod.log | grep waiting
```

> [initandlisten] waiting for connections on port 27017

- **Begin using MongoDB**

```bash
mongo
```

### Useful commands

- **Stop MongoDB**

```bash
sudo service mongod stop
```

- **Restart MongoDB**

```bash
sudo service mongod restart
```
