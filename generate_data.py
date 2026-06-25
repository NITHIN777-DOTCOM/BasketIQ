import os
import random
import numpy as np
import pandas as pd
from faker import Faker

def generate_dataset(num_rows=50000, output_path='data/transactions.csv', start_date_str='2023-01-01', end_date_str='2024-12-31'):
    """
    Generate a realistic synthetic transaction dataset matching UCI Online Retail patterns.
    """
    print(f"Initializing synthetic transaction generator for {num_rows} rows...")
    
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    fake = Faker()
    fake.seed_instance(42) 
    np.random.seed(42)
    random.seed(42)
    
    print("Generating 500 unique customer profiles...")
    customer_ids = sorted(list(set(fake.random_int(min=12000, max=18999) for _ in range(1000))))[:500]
    while len(customer_ids) < 500:
        val = fake.random_int(min=12000, max=18999)
        if val not in customer_ids:
            customer_ids.append(val)
    customer_ids = sorted(customer_ids)
    
    countries = (
        ['United Kingdom'] * 440 + 
        ['Germany'] * 20 + 
        ['France'] * 20 + 
        ['EIRE'] * 10 + 
        ['Spain'] * 10
    )
    random.shuffle(countries)
    customer_country_map = {cid: country for cid, country in zip(customer_ids, countries)}
    
    # 2. Generate 200 unique products
    print("Generating 200 unique products with pricing and stock codes...")
    adjectives = ["ORGANIC", "FRESH", "PREMIUM", "VINTAGE", "LARGE", "MINI", "GARDEN", "HOME", "RETRO", "CLASSIC", "ROYAL", "LUXURY"]
    colors = ["RED", "BLUE", "GREEN", "WHITE", "PINK", "YELLOW", "BLACK", "SILVER", "GOLD", "GREY"]
    base_items = [
        "T-LIGHT HOLDER", "CERAMIC MUG", "SCENTED CANDLE", "COASTER SET", "WALL CLOCK", 
        "WOODEN PICTURE FRAME", "COTTON TOWEL", "TEA TOWEL", "LUNCH BAG", "SHOPPING BAG",
        "BANANAS", "STRAWBERRIES", "WHOLE MILK 1L", "CHEDDAR CHEESE", "FREE RANGE EGGS",
        "SOURDOUGH BREAD", "BUTTER 250G", "GREEK YOGURT", "OLIVE OIL 500ML", "SPAGHETTI 500G",
        "ROASTED COFFEE BEANS", "EARL GREY TEA", "MINERAL WATER 1.5L", "ORANGE JUICE 1L", "DARK CHOCOLATE",
        "POTATO CHIPS", "BEEF STEAK", "CHICKEN BREAST", "FRESH SALMON", "SPINACH 250G",
        "BROWN RICE", "TOMATO SAUCE", "HONEY JAR", "PEANUT BUTTER", "SEWING KIT",
        "GARDEN SPADE", "FLOWER POT", "WATERING CAN", "DOORMAT", "KITCHEN SCALES",
        "OVEN GLOVES", "APRON", "CERAMIC BOWL", "GLASS TUMBLER", "PAPER NAPKINS",
        "HAND SOAP", "SHAMPOO", "HAIR CONDITIONER", "SHOWER GEL", "TOOTHBRUSH", "TOILET ROLL 4 PACK"
    ]

    # 15 affinity bundles — products in the same group are frequently bought together
    AFFINITY_BUNDLE_BASES = [
        ["SOURDOUGH BREAD", "BUTTER 250G", "WHOLE MILK 1L"],
        ["SHAMPOO", "HAIR CONDITIONER", "SHOWER GEL"],
        ["SPAGHETTI 500G", "TOMATO SAUCE", "OLIVE OIL 500ML"],
        ["ROASTED COFFEE BEANS", "EARL GREY TEA", "DARK CHOCOLATE"],
        ["BANANAS", "STRAWBERRIES", "ORANGE JUICE 1L"],
        ["BEEF STEAK", "POTATO CHIPS", "SPINACH 250G"],
        ["FREE RANGE EGGS", "BUTTER 250G", "CHEDDAR CHEESE"],
        ["CERAMIC MUG", "COASTER SET", "PAPER NAPKINS"],
        ["GARDEN SPADE", "FLOWER POT", "WATERING CAN"],
        ["T-LIGHT HOLDER", "SCENTED CANDLE", "CERAMIC BOWL"],
        ["TOOTHBRUSH", "TOILET ROLL 4 PACK", "HAND SOAP"],
        ["FRESH SALMON", "BROWN RICE", "SPINACH 250G"],
        ["CHICKEN BREAST", "BROWN RICE", "TOMATO SAUCE"],
        ["POTATO CHIPS", "PEANUT BUTTER", "MINERAL WATER 1.5L"],
        ["GREEK YOGURT", "HONEY JAR", "STRAWBERRIES"],
    ]
    AFFINITY_PICK_PROB = 0.40
    
    products = []
    seen_descriptions = set()
    seen_stock_codes = set()
    
    while len(products) < 200:
        base = random.choice(base_items)
        style = random.choice([1, 2, 3, 4])
        if style == 1:
            desc = f"{random.choice(adjectives)} {random.choice(colors)} {base}"
        elif style == 2:
            desc = f"{random.choice(adjectives)} {base}"
        elif style == 3:
            desc = f"{random.choice(colors)} {base}"
        else:
            desc = base
            
        stock_code = f"{fake.random_int(min=10000, max=99999)}"
        
        if desc not in seen_descriptions and stock_code not in seen_stock_codes:
            seen_descriptions.add(desc)
            seen_stock_codes.add(stock_code)
            
            unit_price = round(float(np.random.lognormal(mean=0.8, sigma=0.8) + 0.40), 2)
            unit_price = min(max(0.40, unit_price), 95.00)  # Bound price to realistic range
            
            products.append({
                'StockCode': stock_code,
                'Description': desc,
                'UnitPrice': unit_price
            })

    def _build_affinity_groups(catalog, bundle_bases):
        """Map each bundle to product indices whose description matches a bundle item."""
        groups = []
        for bases in bundle_bases:
            indices = []
            for i, prod in enumerate(catalog):
                desc = prod["Description"].upper()
                if any(base in desc for base in bases):
                    indices.append(i)
            if indices:
                groups.append(indices)
        return groups

    affinity_groups = _build_affinity_groups(products, AFFINITY_BUNDLE_BASES)
    print(f"Configured {len(AFFINITY_BUNDLE_BASES)} affinity bundles "
          f"({len(affinity_groups)} with matching products, "
          f"{AFFINITY_PICK_PROB:.0%} invoice affinity rate)")
    
    product_probs = np.exp(-np.linspace(0, 3.5, 200))
    product_probs /= product_probs.sum()
    
    print("Setting up date range and seasonal patterns...")
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    all_dates = pd.date_range(start_date, end_date)
    
    month_multipliers = {
        1: 0.75, 2: 0.80, 3: 0.90, 4: 1.00, 5: 1.05, 6: 1.10,
        7: 1.05, 8: 1.10, 9: 1.15, 10: 1.25, 11: 1.40, 12: 1.60
    }
    day_multipliers = {
        0: 0.85,  # Monday
        1: 0.85,  # Tuesday
        2: 0.90,  # Wednesday
        3: 0.95,  # Thursday
        4: 1.15,  # Friday
        5: 1.45,  # Saturday
        6: 1.30   # Sunday
    }
    
    date_weights = []
    for d in all_dates:
        m_mult = month_multipliers[d.month]
        w_mult = day_multipliers[d.dayofweek]
        date_weights.append(m_mult * w_mult)
        
    date_weights = np.array(date_weights)
    date_weights /= date_weights.sum()
    
    hour_weights = np.array([
        0.001, 0.001, 0.001, 0.001, 0.001, 0.005,  # 00:00 - 05:59 (mostly closed)
        0.010, 0.020, 0.040, 0.060,              # 06:00 - 09:59 (morning opening)
        0.120, 0.120, 0.120, 0.120,              # 10:00 - 13:59 (Peak 1: 10am - 2pm)
        0.060, 0.060, 0.060,                     # 14:00 - 16:59 (mid-afternoon lull)
        0.100, 0.100, 0.100,                     # 17:00 - 19:59 (Peak 2: 5pm - 8pm)
        0.040, 0.020, 0.010, 0.002              
    ])
    hour_weights /= hour_weights.sum()
    
    print("Generating transaction lines (invoices)...")
    invoice_no = 536365
    rows = []
    
    last_reported = 0
    
    while len(rows) < num_rows:
        date_base = np.random.choice(all_dates, p=date_weights)
        hour = np.random.choice(np.arange(24), p=hour_weights)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        invoice_date = pd.Timestamp(date_base).replace(hour=hour, minute=minute, second=second)
        
        cust_id = int(np.random.choice(customer_ids))
        country = customer_country_map[cust_id]
        
        num_items = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
        )
        
        if len(rows) + num_items > num_rows:
            num_items = num_rows - len(rows)

        use_affinity = affinity_groups and random.random() < AFFINITY_PICK_PROB
        if use_affinity:
            group = random.choice(affinity_groups)
            replace = len(group) < num_items
            invoice_prods_indices = np.random.choice(group, size=num_items, replace=replace)
        else:
            invoice_prods_indices = np.random.choice(
                np.arange(200), size=num_items, replace=False, p=product_probs
            )
        
        for idx in invoice_prods_indices:
            prod = products[idx]
            unit_price = prod['UnitPrice']
            
            if unit_price > 15.0:
                q = int(np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03]))
            elif unit_price >= 5.0:
                q = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.60, 0.20, 0.10, 0.05, 0.03, 0.02]))
            else:
                q = int(np.random.choice([1, 2, 3, 4, 6, 8, 10, 12, 24], p=[0.40, 0.20, 0.10, 0.10, 0.05, 0.05, 0.04, 0.04, 0.02]))
                
            rows.append({
                'InvoiceNo': str(invoice_no),
                'StockCode': prod['StockCode'],
                'Description': prod['Description'],
                'Quantity': q,
                'InvoiceDate': invoice_date.strftime('%Y-%m-%d %H:%M:%S'),
                'UnitPrice': unit_price,
                'CustomerID': cust_id,
                'Country': country
            })
            
        invoice_no += 1
        
        current_len = len(rows)
        if current_len // 10000 > last_reported:
            last_reported = current_len // 10000
            print(f"  Generated {current_len:,} / {num_rows:,} rows...")
            
    df = pd.DataFrame(rows)
    
    print(f"Saving dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Generation complete!")
    
    print("\n" + "="*50)
    print("            DATASET VERIFICATION STATS            ")
    print("="*50)
    print(f"Total Rows:             {len(df):,}")
    print(f"Unique Invoices:        {df['InvoiceNo'].nunique():,}")
    print(f"Unique Customers:       {df['CustomerID'].nunique()}")
    print(f"Unique Products:        {df['StockCode'].nunique()}")
    print(f"Date Range:             {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Min Unit Price:         ${df['UnitPrice'].min():.2f}")
    print(f"Max Unit Price:         ${df['UnitPrice'].max():.2f}")
    print(f"Avg Unit Price:         ${df['UnitPrice'].mean():.2f}")
    
    df['dt'] = pd.to_datetime(df['InvoiceDate'])
    df['Hour'] = df['dt'].dt.hour
    
    peak_1 = df['Hour'].isin([10, 11, 12, 13]).sum()
    peak_2 = df['Hour'].isin([17, 18, 19]).sum()
    total_peak = peak_1 + peak_2
    pct_peak = (total_peak / len(df)) * 100
    
    print(f"Peak Hour (10am-2pm) Rows: {peak_1:,} ({peak_1/len(df)*100:.1f}%)")
    print(f"Peak Hour (5pm-8pm) Rows:  {peak_2:,} ({peak_2/len(df)*100:.1f}%)")
    print(f"Total Peak Hour Rows:   {total_peak:,} ({pct_peak:.1f}%)")
    
    df['Month'] = df['dt'].dt.month
    nov_dec = df['Month'].isin([11, 12]).sum()
    jan_feb = df['Month'].isin([1, 2]).sum()
    print(f"Holiday Spike (Nov-Dec): {nov_dec:,} ({nov_dec/len(df)*100:.1f}%)")
    print(f"Winter Lull (Jan-Feb):   {jan_feb:,} ({jan_feb/len(df)*100:.1f}%)")

    print("\nCountry Distribution:")
    country_counts = df.groupby('Country')['InvoiceNo'].nunique()
    for country, count in country_counts.items():
        print(f"  - {country}: {count:,} invoices")
    print("="*50 + "\n")

if __name__ == '__main__':
    generate_dataset()
