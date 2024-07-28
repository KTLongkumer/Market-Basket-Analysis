#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Provide the correct path to your CSV file
file_path = 'basket_analysis.csv'  # Replace with the correct path if necessary

# Load the dataset
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())


# In[5]:


from mlxtend.frequent_patterns import apriori, association_rules

# Generate frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the top rules
print(rules.head())


# In[6]:


import matplotlib.pyplot as plt

# Scatter plot for support vs. confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs. Confidence')
plt.show()


# In[7]:


# Top 10 rules by lift
top_rules = rules.nlargest(10, 'lift')

# Bar plot
plt.barh(top_rules.index, top_rules['lift'], color='skyblue')
plt.xlabel('Lift')
plt.title('Top 10 Rules by Lift')
plt.show()


# In[8]:


import networkx as nx

# Create a graph
G = nx.DiGraph()

# Add nodes and edges from rules
for _, rule in top_rules.iterrows():
    G.add_node(frozenset(rule['antecedents']), size=rule['support'])
    G.add_node(frozenset(rule['consequents']), size=rule['support'])
    G.add_edge(frozenset(rule['antecedents']), frozenset(rule['consequents']), weight=rule['lift'])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=[G.nodes[node]['size']*1000 for node in G])
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Network Graph of Association Rules')
plt.show()


# In[9]:


# Display top rules by confidence and lift
top_confidence_rules = rules.nlargest(10, 'confidence')
top_lift_rules = rules.nlargest(10, 'lift')

print("Top 10 Rules by Confidence:")
print(top_confidence_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\nTop 10 Rules by Lift:")
print(top_lift_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Actionable Insights

# Example of analyzing top rule by confidence
for index, rule in top_confidence_rules.iterrows():
    antecedents = list(rule['antecedents'])
    consequents = list(rule['consequents'])
    support = rule['support']
    confidence = rule['confidence']
    lift = rule['lift']

    print(f"\nRule: {antecedents} -> {consequents}")
    print(f"Support: {support}")
    print(f"Confidence: {confidence}")
    print(f"Lift: {lift}")

    # Actionable Insight: Product Placement
    print(f"Recommendation: Place {antecedents} and {consequents} near each other in the store to increase visibility and convenience for customers.")

# Example of analyzing top rule by lift
for index, rule in top_lift_rules.iterrows():
    antecedents = list(rule['antecedents'])
    consequents = list(rule['consequents'])
    support = rule['support']
    confidence = rule['confidence']
    lift = rule['lift']

    print(f"\nRule: {antecedents} -> {consequents}")
    print(f"Support: {support}")
    print(f"Confidence: {confidence}")
    print(f"Lift: {lift}")

    # Actionable Insight: Promotional Strategy
    print(f"Recommendation: Design a promotion offering a discount on {consequents} when customers purchase {antecedents}.")


# In[ ]:




