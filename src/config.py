from datetime import date, timedelta


symbol    = 'BOSCHLTD'
N         = 15000 
startDate = date.today() - timedelta(N)
endDate   = date.today()

train_per = 0.70
test_per  = 0.15
valid_per  = 0.15