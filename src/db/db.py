import sqlite3
# TODO process data & pass to insert
def deletePoloniex():
    connection.execute('''DROP TABLE POLONIEX;''')
def buildPoloniex():
    connection = sqlite3.connect('crypto.db')
    print("Opened database")
    connection.execute('''CREATE TABLE POLONIEX
        (ID INT PRIMARY KEY NOT NULL,
        COIN TEXT NOT NULL,
        PRICEHIGH NUMERIC,
        PRICELOW NUMERIC,
        PRICEOPEN NUMERIC,
        PRICECLOSE NUMERIC,
        VOLUME NUMERIC,
        QUOTEVOLUME NUMERIC,
        DATETIME DATETIME NOT NULL,
        PRICEWEIGHTEDAVERAGE NUMERIC);''')
    print("Table Created succesfully")
    connection.close()
    
def insertPoloniex():
    connection = sqlite3.connect('crypto.db')
    connection.execute("""INSERT INTO POLONIEX ('ID','COIN', 'VOLUME', 'DATETIME')
        VALUES (1,'TEST', 420, 30000000000 )""");
    connection.commit()
    print("Record created test")
    connection.close()

def checkPoloniex():
    connection = sqlite3.connect('crypto.db')
    cursor = connection.execute('''SELECT * FROM POLONIEX''')
    print(cursor)
    for row in cursor:
        print(row)
    connection.close()

