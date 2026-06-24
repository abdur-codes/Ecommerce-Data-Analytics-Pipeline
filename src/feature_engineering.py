import pandas as pd
import numpy as np


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['order_year'] = df['Date'].dt.year
        df['order_month'] = df['Date'].dt.month
        df['order_day'] = df['Date'].dt.day
        df['order_dayofweek'] = df['Date'].dt.dayofweek
        df['order_quarter'] = df['Date'].dt.quarter
        df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)
        print("  Feature: 'order_year', 'order_month', 'order_day', 'order_dayofweek', 'order_quarter', 'is_weekend'")
    return df


def create_coupon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'CouponCode' in df.columns:
        df['had_coupon'] = (df['CouponCode'] != 'None').astype(int)
        df['coupon_type'] = df['CouponCode'].apply(
            lambda x: x if x != 'None' else 'NO_COUPON'
        )
        print("  Feature: 'had_coupon' (whether coupon was used)")
        print("  Feature: 'coupon_type' (normalized coupon label)")
    return df


def create_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'TotalPrice' in df.columns and 'Quantity' in df.columns:
        df['avg_item_price'] = df['TotalPrice'] / df['Quantity']
        df['avg_item_price'] = df['avg_item_price'].replace([np.inf, -np.inf], np.nan).fillna(0)
        print("  Feature: 'avg_item_price' (TotalPrice / Quantity)")
    if 'TotalPrice' in df.columns and 'ItemsInCart' in df.columns:
        df['price_per_cart_item'] = df['TotalPrice'] / df['ItemsInCart']
        df['price_per_cart_item'] = df['price_per_cart_item'].replace([np.inf, -np.inf], np.nan).fillna(0)
        print("  Feature: 'price_per_cart_item' (TotalPrice / ItemsInCart)")
    if 'Quantity' in df.columns and 'ItemsInCart' in df.columns:
        df['cart_fill_rate'] = (df['Quantity'] / df['ItemsInCart'] * 100).round(2)
        print("  Feature: 'cart_fill_rate' (Quantity / ItemsInCart * 100)")
    return df


def create_shipping_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'ShippingAddress' in df.columns:
        df['address_type'] = df['ShippingAddress'].str.extract(r'(\d+)\s+(.*)', expand=False)[1]
        df['address_type'] = df['address_type'].fillna('Unknown')
        print("  Feature: 'address_type' (extracted from ShippingAddress)")
    return df


def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'PaymentMethod' in df.columns:
        is_card = df['PaymentMethod'].str.contains('Card', case=False, na=False)
        df['is_card_payment'] = is_card.astype(int)
        print("  Feature: 'is_card_payment' (Credit/Debit Card indicator)")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['quantity_unitprice_interaction'] = df['Quantity'] * df['UnitPrice']
        print("  Feature: 'quantity_unitprice_interaction' (Quantity * UnitPrice)")
    if 'order_month' in df.columns and 'had_coupon' in df.columns:
        df['month_coupon_interaction'] = df['order_month'] * df['had_coupon']
        print("  Feature: 'month_coupon_interaction' (order_month * had_coupon)")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = create_date_features(df)
    df = create_coupon_features(df)
    df = create_revenue_features(df)
    df = create_shipping_features(df)
    df = create_payment_features(df)
    df = create_interaction_features(df)

    print(f"\n[FE] Created 10+ new features.")
    print(f"  Shape after feature engineering: {df.shape}")
    return df
