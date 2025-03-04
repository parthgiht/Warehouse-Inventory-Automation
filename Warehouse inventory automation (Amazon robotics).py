#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math


# ## EOQ & ROP model

# In[6]:


# Function to calculate EOQ
def calculate_eoq(demand, ordering_cost, holding_cost):
    return math.sqrt((2 * demand * ordering_cost) / holding_cost)


# Function to calculate the number of orders per year
def calculate_number_of_orders(demand, eoq):
    return demand / eoq


# Function to calculate reorder point
def calculate_reorder_point(annual_demand, lead_time, working_days_per_year=250):
    daily_demand = annual_demand / working_days_per_year
    return daily_demand * lead_time


# Import CSV file
df = pd.read_csv('Warehouse_Inventory_Automation.csv')  


# Calculate total annual demand 
total_annual_demand = df['Daily_Sales'].sum() * 365  


# ordering and holding costs
total_ordering_cost = df['Restock_Cost'].sum()  
total_holding_cost = df['Storage_Cost'].sum() 


# Calculate overall EOQ 
eoq = calculate_eoq(total_annual_demand, total_ordering_cost, total_holding_cost)


# Calculate overall number of orders per year
num_orders = calculate_number_of_orders(total_annual_demand, eoq)


# Calculate overall reorder point using the average lead time
average_lead_time = df['Lead_Time'].mean() 
reorder_point = calculate_reorder_point(total_annual_demand, average_lead_time)


# results
print(df.head())
print("\nOverall EOQ and Reorder Point Results")
print("--------------------------------------------------")
print(f"Total Annual Demand :- [{total_annual_demand} units]")
print(f"Economic Order Quantity (EOQ) :- [{eoq:.2f} units]")
print(f"Number of Orders per Year :- [{num_orders:.0f} orders]")
print(f"Reorder Point (ROP) :- [{reorder_point:.0f} units]")


# In[ ]:





# ## Automation Setup 
# - Minimize total inventory cost by implementing automation in warehouse 

# In[20]:


# Read the CSV file
df = pd.read_csv('Warehouse_Inventory_Automation.csv')

def robotics_eoq_model():
    # Calculate EOQ considering robotics setup costs
    setup_cost_factor = 1.2  # 20% increase due to robotics
    
    df['Robotics_EOQ'] = np.sqrt(
        (2 * df['Daily_Sales'] * 365 * df['Restock_Cost'] * setup_cost_factor) / 
        df['Storage_Cost']
    ).round(2)
    
    # final results
    results = pd.DataFrame({
        'Item_ID': df['Item_ID'],
        'Category': df['Category'],
        'Optimized_Order_Quantity': df['Robotics_EOQ']
    })
    
    # Display results
    print("\nRobotics EOQ Results:")
    print("-" * 50)
    print(results)
    
    return results
    
if __name__ == "__main__":
    result_df = robotics_eoq_model()
    
    
# Plot the result    
plt.figure(figsize=(10, 6))
plt.bar(result_df['Category'], result_df['Optimized_Order_Quantity'], color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Optimized Order Quantity')
plt.title('Optimized Order Quantity for Each Item (Robotics)')
plt.xticks(rotation=45) 
plt.legend(['Automation EOQ']) 
plt.grid(True)
plt.show()


# ## Automation Maintenance 
# - Avoiding stockouts during robot maintenance (Safety stock)

# In[4]:


df = pd.read_csv('Warehouse_Inventory_Automation.csv')

def maintenance_eoq_model():
    # Consider maintenance downtime in calculations
    working_days = 353  # Assuming 12 days maintenance per year
    
    # Calculate basic EOQ
    df['EOQ'] = np.sqrt(
        (2 * df['Daily_Sales'] * working_days * df['Restock_Cost']) / 
        df['Storage_Cost']
    )
    
    # Add safety stock based on lead time
    df['Safety_Stock'] = df['Lead_Time'] * df['Daily_Sales']
    df['Maintenance_EOQ'] = (df['EOQ'] + df['Safety_Stock']).round(2)
    
    # Create final results
    results = pd.DataFrame({
        'Item_ID': df['Item_ID'],
        'Category': df['Category'],
        'Order_during_Maintenance': df['Maintenance_EOQ']
    })
    
    # Display results
    print("\nOrder_during_Maintenance EOQ Results:")
    print("-" * 50)
    print(results)
    
    return results

if __name__ == "__main__":
    results_df = maintenance_eoq_model()
    
category_totals = results_df.groupby('Category')['Order_during_Maintenance'].sum()


colors = plt.cm.Paired.colors[:len(category_totals)]
plt.figure(figsize=(8, 6))
category_totals.plot(kind='bar', color=colors)
plt.xlabel('Categories')
plt.ylabel('Total Order Quantity During Maintenance')
plt.title('Order Quantity by Category')
plt.xticks(rotation=45)
plt.legend(['Total Order Quantity'])
plt.show()


# ## Automation Space-requirement 
# - Space requirement for automation can help to manage inventory efficiently.

# In[5]:


df = pd.read_csv('Warehouse_Inventory_Automation.csv')

def space_constrained_eoq_model():
    # Basic EOQ with space utilization factor
    space_factor = 0.85  # 85% space utilization
    
    # Calculate basic EOQ
    df['EOQ'] = np.sqrt(
        (2 * df['Daily_Sales'] * 365 * df['Restock_Cost']) / 
        df['Storage_Cost']
    ) * space_factor
    
    # Adjust for automation requirements
    df['Space_EOQ'] = np.where(
        df['Automation_Required'] == True,
        df['EOQ'] * 0.9,  # Reduce by 9% to 10% if automation required
        df['EOQ']
    ).round(2)
    
    # Create final results
    results = pd.DataFrame({
        'Item_ID': df['Item_ID'],
        'Category': df['Category'],
        'Space_for_automation': df['Space_EOQ']
    })
    
    # Display results
    print("\nSpace-requirement EOQ Results:")
    print("-" * 50)
    print(results)
    
    # Plot the results
    colors = plt.cm.Paired.colors[:len(category_totals)]
    plt.figure(figsize=(10, 6))
    plt.bar(results['Category'], results['Space_for_automation'], color=colors, edgecolor='black')
    plt.title('Space Requirement by Item (Automation)', fontsize=14)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Space for Automation (units)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(['Space requirement'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    space_constrained_eoq_model()


# In[ ]:




