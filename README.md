# Signed-PageRank: Influence Maximization in Signed Social Networks

Based on the paper: *"Signed-PageRank: An Efficient Influence Maximization Framework for Signed Social Networks"* (IEEE TKDE 2021) by Xiaoyan Yin et al.

---

## 1. The Big Picture

Imagine you are a marketer launching a new product. You have a limited budget, so you can only give free samples to a few key people (we call them **"seed nodes"**). Your goal is to select the absolute best individuals so that through word-of-mouth, the maximum number of people end up buying your product.

In the past, researchers studied this problem (called **Influence Maximization**) by looking at social networks where all links were positive (like being "friends" on Facebook). But in the real world, relationships are more complex. Sometimes people trust each other (+1), and sometimes they distrust or dislike each other (-1). Networks that include both positive and negative relationships are called **Signed Social Networks**.

This paper tackles a fascinating problem: **How does information spread when enemies are involved?** 
If your best friend recommends a movie, you'll probably like it. But if someone you heavily distrust recommends the same movie, you might intentionally avoid it! The authors developed a framework to model this behavior and designed an algorithm (**Signed-PageRank**) to efficiently find the best seed nodes in such complex networks.

---

## 2. How Do People Make Decisions? (Beliefs vs. Attitudes)

To model human psychology accurately, the framework separates *how you feel* from *what you do*:

- **Belief ($x_i$):** A continuous number between 0 and 1. It represents your internal likelihood of liking the product. If your belief is 0.8, you are 80% convinced.
- **Attitude:** A binary choice (0 or 1). This is whether you actually accept the information (e.g., buy the product or share the post). Your attitude is drawn probabilistically based on your belief.

Furthermore, people aren’t paying attention forever. Every individual has a **Recommendation Cycle**—a specific time window during which they actually care about recommendations. If you see the ad too early or too late, you ignore it.

---

## 3. The Tug-of-War: Updating Beliefs

When a person is in their recommendation cycle, they look at their neighbors who have already accepted the information to update their own belief. The update is a tug-of-war influenced by two factors:

1. **Positive Embeddedness ($\alpha$):** The proportion of your incoming links that are *friends* (positive). 
2. **Negative Embeddedness ($\beta$):** The proportion of your incoming links that are *foes* (negative).

**The Rule:** 
* Your belief is **pulled toward** the beliefs of your friends. (If your friend likes it more than you do, your belief goes up).
* Your belief is **pushed away** from the beliefs of your foes. (If your enemy likes it more than you do, your belief goes down). 

This dynamic equation ensures that the network continuously balances trust and distrust as information propagates.

---

## 4. The Algorithm: Signed-PageRank (SPR)

Simulating this tug-of-war across millions of people to find the perfect starting point is computationally exhausting. To solve this, the authors invented **Signed-PageRank (SPR)**, inspired by how Google originally ranked webpages.

Instead of running slow simulations, SPR mathematically calculates the "Rank" (Influence power) of every node based on the network structure:

1. **The SPR Matrix:** It builds a matrix that combines the weight of the connections with their signs (+1 or -1).
2. **The Flow of Influence:** In traditional PageRank, a node shares its rank with outward neighbors. In Signed-PageRank, the algorithm updates a node's score by looking at the *differences* between its score and its outgoing neighbors' scores over both positive and negative edges. 
3. **Efficiency:** Because it uses vector/matrix math, the algorithm converges rapidly (typically in fewer iterations than standard random simulation), turning an extremely complex psychological simulation into a fast mathematical operation.

By picking the `k` nodes with the highest Signed-PageRank scores, we get a group of seed nodes that are mathematically positioned to overwhelm negative resistance and maximize positive cascade.

---

## 5. About the Implementation Code

The provided `signed_pagerank.py` file rigorously implements the entire paper:

* **Information Propagation Framework**: It accurately simulates the continuous belief updates, discrete attitude conversions, and time-bound recommendation cycles.
* **SPR Algorithm**: It implements both an iterative array-based version and a fast matrix-vectorised calculation for the $O(N)$ Signed-PageRank algorithm.

### Running the Code

```bash
python signed_pagerank.py --dataset synthetic --k 10
python signed_pagerank.py --dataset epinions --k 20 --max-nodes 3000
python signed_pagerank.py --dataset slashdot --k 15
python signed_pagerank.py --dataset wiki --k 10
```
