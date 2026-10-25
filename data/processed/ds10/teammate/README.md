README file for training the model using Bitcoin dataset (does not contain any information about banking dataset)

20-30 addresses with at least 50 confirmed transactions over at least 90 days, plus 2-3 short-history ones.

data/
	transactions.csv (one row per confirmed transaction per address): Explains the chronological transaction ledgers in its unprocessed form.
		address: the Bitcoin address the row describes.
		txid: the transaction ID. The same ID can appear in rows for different addresses.
		block_height: the block that confirmed the transaction.
		confirmed_utc: the block time in UTC, as readable text.
		block_time: the same time as UTC, handy for sorting and time calculations.
		received_btc: the sum of the transaction’s outputs that pay this address, in BTC.
		sent_btc: the sum of the inputs spent from this address, in BTC.
		net_btc: received minus sent. Positive means “in” and negative means “out”. When the address is the sender, this includes the transaction fee, because the fee is part of what left the address.
		direction: in, out, or none when net is zero.

	address_summary.csv (one row per address): Details the high-level metrics (confirmed transaction counts, span in days) used to filter long-history training addresses versus short-history contrast addresses.
		address: the Bitcoin address.
		group: low (50-149 transactions), medium (150-299), high (300-600), or short_history (4-9).
		address_type: an approximate type from the address prefix (legacy, SegWit, Taproot or script).
		confirmed_tx_count: the number of confirmed transactions retrieved.
		first_tx_utc and last_tx_utc: the first and last confirmed times.
		span_days: the days between the first and last transaction.

usage/
	1. make sure both Anaconda/Miniconda and VSC is downloaded
	2. make sure user Anaconda environment is active
	3. make sure the default profile in VSC is set to Anaconda
	4. run the Inno-Assign2.py
	5. data file containing the retrieved dataset should appear within the environment cd

notes/
	all times are recorded in UTC
	bitcoin intake are recorded in Satoshi(sat) : 0.00000001 BTC(the smallest possible unit, spanning 8 decimal places)