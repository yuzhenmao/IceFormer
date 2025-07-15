/*
 * Code for IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs
 * 
 * This code implements the method described in the IceFormer paper, 
 * which can be found at https://openreview.net/forum?id=6RR3wU4mSZ
 * 
 * This file is a part of the Dynamic Continuous Indexing reference 
 * implementation.
 * 
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. 
 * 
 * Copyright (C) 2024    Yuzhen Mao, Ke Li
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "btree_p.h"

// Private helper functions
static bool
btree_p_insert_descend(btree_p *const tree, void *n, const float key, data_pt const value, float *splitkey,
                       void **splitnode);

static void btree_p_split_leaf(btree_p *const tree, btree_p_leaf_node *leaf, float *_newkey, void **_newleaf);

static void btree_p_split_inner(btree_p *const tree, btree_p_inner_node *inner, float *_newkey, void **_newinner,
                                unsigned int addslot);

static int btree_p_find_lower_inner(const btree_p_inner_node *n, const float key);

static int btree_p_find_lower_leaf(const btree_p_leaf_node *n, const float key);

static void btree_p_free_node(void *n);

static bool btree_p_is_leaf(const void *n);

static unsigned short btree_p_num_slots_used(void *n);

static float *btree_p_copy_keys_fwd(float *first, float *last, float *d_first);

static float *btree_p_copy_keys_bwd(float *first, float *last, float *d_last);

static data_pt *btree_p_copy_data_fwd(data_pt *first, data_pt *last, data_pt *d_first);

static data_pt *btree_p_copy_data_bwd(data_pt *first, data_pt *last, data_pt *d_last);

static void **btree_p_copy_child_fwd(void **first, void **last, void **d_first);

static void **btree_p_copy_child_bwd(void **first, void **last, void **d_last);

static btree_c_result
btree_p_delete_descend(btree_p *const tree, const float key, const long long value, void *curr, void *left,
                       void *right, btree_p_inner_node *leftparent, btree_p_inner_node *rightparent,
                       btree_p_inner_node *parent, unsigned int parentslot);

static btree_c_result btree_p_merge_leaves(btree_p *const tree, btree_p_leaf_node *left, btree_p_leaf_node *right,
                                           btree_p_inner_node *parent);

static btree_c_result
btree_p_merge_inner(btree_p_inner_node *left, btree_p_inner_node *right, btree_p_inner_node *parent,
                    unsigned int parentslot);

static btree_c_result
btree_p_shift_left_leaf(btree_p_leaf_node *left, btree_p_leaf_node *right, btree_p_inner_node *parent,
                        unsigned int parentslot);

static void btree_p_shift_left_inner(btree_p_inner_node *left, btree_p_inner_node *right, btree_p_inner_node *parent,
                                     unsigned int parentslot);

static void btree_p_shift_right_leaf(btree_p_leaf_node *left, btree_p_leaf_node *right, btree_p_inner_node *parent,
                                     unsigned int parentslot);

static void btree_p_shift_right_inner(btree_p_inner_node *left, btree_p_inner_node *right, btree_p_inner_node *parent,
                                      unsigned int parentslot);

void btree_p_init(btree_p *tree) {
    tree->num_data = 0;
    tree->leaf_max_num_slots = BTREE_LEAF_MAX_NUM_SLOTS;
    tree->inner_max_num_slots = BTREE_INNER_MAX_NUM_SLOTS;
    tree->leaf_min_num_slots = (tree->leaf_max_num_slots / 2);
    tree->inner_min_num_slots = (tree->inner_max_num_slots / 2);
    tree->root = NULL;
    tree->first_leaf = NULL;
    tree->last_leaf = NULL;
}

// Allocate and initialize a leaf node
static btree_p_leaf_node *btree_p_allocate_leaf() {
    btree_p_leaf_node *n = (btree_p_leaf_node *) malloc(sizeof(btree_p_leaf_node));
    n->prev_leaf = n->next_leaf = NULL;
    n->level = 0;
    n->num_slots_used = 0;
    n->slot_keys = malloc(sizeof(float) * BTREE_LEAF_MAX_NUM_SLOTS);
    n->slot_data = malloc(sizeof(data_pt) * BTREE_LEAF_MAX_NUM_SLOTS);
    return n;
}

// Allocate and initialize an inner node
static btree_p_inner_node *btree_p_allocate_inner(unsigned short level) {
    btree_p_inner_node *n = (btree_p_inner_node *) malloc(sizeof(btree_p_inner_node));
    n->level = level;
    n->num_slots_used = 0;
    n->slot_keys = malloc(sizeof(float) * BTREE_INNER_MAX_NUM_SLOTS);
    n->slot_ptrs = malloc(sizeof(void *) * (BTREE_INNER_MAX_NUM_SLOTS + 1));
    return n;
}

// True if this is a leaf node
static bool btree_p_is_leaf(const void *n) {
    return (*(unsigned short *) n == 0);
}

// True if node has too few entries
static bool btree_p_is_below_min(const btree_p *const tree, const void *n) {
    if (btree_p_is_leaf(n)) {
        btree_p_leaf_node *nn = (btree_p_leaf_node *) n;
        return (nn->num_slots_used < tree->leaf_min_num_slots);
    } else {
        btree_p_inner_node *nn = (btree_p_inner_node *) n;
        return (nn->num_slots_used < tree->inner_min_num_slots);
    }
}

// True if few used entries, less than half full
static bool btree_p_is_few(const btree_p *const tree, const void *n) {
    if (btree_p_is_leaf(n)) {
        btree_p_leaf_node *nn = (btree_p_leaf_node *) n;
        return (nn->num_slots_used <= tree->leaf_min_num_slots);
    } else {
        btree_p_inner_node *nn = (btree_p_inner_node *) n;
        return (nn->num_slots_used <= tree->inner_min_num_slots);
    }
}

// Return the num_slots_used component of leafnode or innernode
static unsigned short btree_p_num_slots_used(void *n) {
    if (btree_p_is_leaf(n)) {
        btree_p_leaf_node *ln = (btree_p_leaf_node *) (n);
        return ln->num_slots_used;
    } else {
        btree_p_inner_node *in = (btree_p_inner_node *) (n);
        return in->num_slots_used;
    }
}

// Correctly free either inner or leaf node, destructs all contained key
// and value objects
static void btree_p_free_node(void *n) {
    if (btree_p_is_leaf(n)) {
        btree_p_leaf_node *ln = (btree_p_leaf_node *) (n);
        free(ln->slot_keys);
        free(ln->slot_data);
        free(ln);
    } else {
        btree_p_inner_node *in = (btree_p_inner_node *) (n);
        free(in->slot_keys);
        free(in->slot_ptrs);
        free(in);
    }
}
/* 
 * Fast Destruction of the B+ Tree
 */

// Recursively free up nodes
static void clear_recursive(void *n) {
    if (btree_p_is_leaf(n)) {
        // leaf_node itself will be freed by its id
    } else {
        btree_p_inner_node *innernode = (btree_p_inner_node *) (n);

        for (unsigned short slot = 0; slot < innernode->num_slots_used + 1; ++slot) {
            clear_recursive(innernode->slot_ptrs[slot]);
            btree_p_free_node(innernode->slot_ptrs[slot]);
        }
    }
}

// Frees all key/data pairs and all nodes of the tree
void btree_p_clear(btree_p *tree) {
    if (tree->root) {
        clear_recursive(tree->root);
        btree_p_free_node(tree->root);

        tree->root = NULL;
        tree->first_leaf = tree->last_leaf = NULL;
        tree->num_data = 0;
    }

}

// Set the (key,data) pair in slot. Function used by
// bulk_load().
static void btree_p_set_slot(btree_p_leaf_node *n, unsigned short slot, const float key, const data_pt value) {
    assert(slot < n->num_slots_used);
    n->slot_keys[slot] = key;
    n->slot_data[slot] = value;
}

// True if the node's slots are full
static bool btree_p_is_full(const btree_p *const tree, const void *n) {
    if (btree_p_is_leaf(n)) {
        btree_p_leaf_node *nn = (btree_p_leaf_node *) n;
        return (nn->num_slots_used == tree->leaf_max_num_slots);
    } else {
        btree_p_inner_node *nn = (btree_p_inner_node *) n;
        return (nn->num_slots_used == tree->inner_max_num_slots);
    }
}

// Copy keys in forward order
static float *btree_p_copy_keys_fwd(float *first, float *last, float *d_first) {
    while (first != last) {
        *d_first++ = *first++;
    }
    return d_first;
}

// Copy data in forward order
static data_pt *btree_p_copy_data_fwd(data_pt *first, data_pt *last, data_pt *d_first) {
    while (first != last) {
        *d_first++ = *first++;
    }
    return d_first;
}

// Copy keys in reverse order
static float *btree_p_copy_keys_bwd(float *first, float *last, float *d_last) {
    while (first != last) {
        *(--d_last) = *(--last);
    }
    return d_last;
}

// Copy data in reverse order
static data_pt *btree_p_copy_data_bwd(data_pt *first, data_pt *last, data_pt *d_last) {
    while (first != last) {
        *(--d_last) = *(--last);
    }
    return d_last;
}

// Copy child in reverse order
static void **btree_p_copy_child_bwd(void **first, void **last, void **d_last) {
    while (first != last) {
        *(--d_last) = *(--last);
    }
    return d_last;
}

// Copy child in forward order
static void **btree_p_copy_child_fwd(void **first, void **last, void **d_first) {
    while (first != last) {
        *d_first++ = *first++;
    }
    return d_first;
}

/* 
 * B+ Tree Node Binary Search Functions
 */

// Searches for the first key in the node n greater or equal to key. Uses
// binary search with an optional linear self-verification. This is not a
// template function, because the slot_keys array is located at different
// places in btree_p_leaf_node and btree_p_inner_node.
static int btree_p_find_lower_inner(const btree_p_inner_node *n, const float key) {
    if (sizeof(n->slot_keys) > BTREE_BIN_SEARCH_THRESH) {
        if (n->num_slots_used == 0) return 0;

        int lo = 0, hi = n->num_slots_used;

        while (lo < hi) {
            int mid = (lo + hi) >> 1;

            if (key <= n->slot_keys[mid]) {
                hi = mid; // key <= mid
            } else {
                lo = mid + 1; // key > mid
            }
        }

        return lo;
    } else // for nodes <= BTREE_BIN_SEARCH_THRESH do linear search.
    {
        int lo = 0;
        while (lo < n->num_slots_used && n->slot_keys[lo] < key) ++lo;
        return lo;
    }
}

static int btree_p_find_lower_leaf(const btree_p_leaf_node *n, const float key) {
    if (sizeof(n->slot_keys) > BTREE_BIN_SEARCH_THRESH) {
        if (n->num_slots_used == 0) return 0;

        int lo = 0, hi = n->num_slots_used;

        while (lo < hi) {
            int mid = (lo + hi) >> 1;

            if (key <= n->slot_keys[mid]) {
                hi = mid; // key <= mid
            } else {
                lo = mid + 1; // key > mid
            }
        }

        return lo;
    } else // for nodes <= BTREE_BIN_SEARCH_THRESH do linear search.
    {
        int lo = 0;
        while (lo < n->num_slots_used && n->slot_keys[lo] < key) ++lo;
        return lo;
    }
}

// Start the insertion descent at the current root and handle root
// splits. Returns true if the item was inserted
bool btree_p_insert(btree_p *const tree, const float key, const data_pt value) {
    void *newchild = NULL;
    float newkey;

    if (tree->root == NULL) {
        tree->root = tree->first_leaf = tree->last_leaf = btree_p_allocate_leaf();
    }

    bool r = btree_p_insert_descend(tree, tree->root, key, value, &newkey, &newchild);

    if (newchild) {
        btree_p_inner_node *newroot = btree_p_allocate_inner(*(unsigned short *) tree->root + 1);//tree->root->level+1
        newroot->slot_keys[0] = newkey;

        newroot->slot_ptrs[0] = tree->root;
        newroot->slot_ptrs[1] = newchild;

        newroot->num_slots_used = 1;

        tree->root = newroot;
    }

    tree->num_data += 1;
    return r;
}

// Insert a key/value pair into the B+ tree by descending down the nodes to a 
// leaf and inserting the key/data pair in a free slot. If the node overflows, 
// then it must be split and the new split node inserted into the id.
static bool
btree_p_insert_descend(btree_p *const tree, void *n, const float key, const data_pt value, float *splitkey,
                       void **splitnode) {
    if (!btree_p_is_leaf(n)) {
        btree_p_inner_node *inner = (btree_p_inner_node *) (n);

        float newkey;
        void *newchild = NULL;

        int slot = btree_p_find_lower_inner(inner, key);
#ifdef BTREE_DEBUG
        printf("btree::insert_descend into 0x%x\n", inner->slot_ptrs[slot]);
#endif
        bool r = btree_p_insert_descend(tree, inner->slot_ptrs[slot],
                                        key, value, &newkey, &newchild);

        if (newchild) {
#ifdef BTREE_DEBUG
            printf("btree::insert_descend newchild with key %f node 0x%x at slot %d\n" ,newkey, newchild, slot);
#endif
            if (btree_p_is_full(tree, inner)) {
                btree_p_split_inner(tree, inner, splitkey, splitnode, slot);

                if (slot == inner->num_slots_used + 1 && inner->num_slots_used < btree_p_num_slots_used(*splitnode)) {
                    // special case when the insert slot matches the split
                    // place between the two nodes, then the insert key
                    // becomes the split key.

                    assert(inner->num_slots_used + 1 < tree->inner_max_num_slots);

                    btree_p_inner_node *splitinner = (btree_p_inner_node *) (*splitnode);

                    // move the split key and it's datum into the left node
                    inner->slot_keys[inner->num_slots_used] = *splitkey;
                    inner->slot_ptrs[inner->num_slots_used + 1] = splitinner->slot_ptrs[0];
                    inner->num_slots_used++;

                    // set new split key and move corresponding datum into right node
                    splitinner->slot_ptrs[0] = newchild;
                    *splitkey = newkey;

                    return r;
                } else if (slot >= inner->num_slots_used + 1) {
                    // in case the insert slot is in the newly create split
                    // node, we reuse the code below.

                    slot -= inner->num_slots_used + 1;
                    inner = (btree_p_inner_node *) (*splitnode);
                }
            }

            // move items and put pointer to child node into correct slot
            assert(slot >= 0 && slot <= inner->num_slots_used);

            btree_p_copy_keys_bwd(inner->slot_keys + slot, inner->slot_keys + inner->num_slots_used,
                                  inner->slot_keys + inner->num_slots_used + 1);
            btree_p_copy_child_bwd(inner->slot_ptrs + slot, inner->slot_ptrs + inner->num_slots_used + 1,
                                   inner->slot_ptrs + inner->num_slots_used + 2);

            inner->slot_keys[slot] = newkey;
            inner->slot_ptrs[slot + 1] = newchild;
            inner->num_slots_used++;
        }

        return r;
    } else // n->btree_p_is_leaf() == true
    {
        btree_p_leaf_node *leaf = (btree_p_leaf_node *) (n);

        int slot = btree_p_find_lower_leaf(leaf, key);

        if (!allow_duplicates && slot < leaf->num_slots_used && key == leaf->slot_keys[slot]) {
            return false;
        }

        if (btree_p_is_full(tree, leaf)) {
            btree_p_split_leaf(tree, leaf, splitkey, splitnode);

            // check if insert slot is in the split sibling node
            if (slot >= leaf->num_slots_used) {
                slot -= leaf->num_slots_used;
                leaf = (btree_p_leaf_node *) (*splitnode);
            }
        }

        // move items and put data item into correct data slot
        assert(slot >= 0 && slot <= leaf->num_slots_used);

        btree_p_copy_keys_bwd(leaf->slot_keys + slot, leaf->slot_keys + leaf->num_slots_used,
                              leaf->slot_keys + leaf->num_slots_used + 1);
        btree_p_copy_data_bwd(leaf->slot_data + slot, leaf->slot_data + leaf->num_slots_used,
                              leaf->slot_data + leaf->num_slots_used + 1);

        leaf->slot_keys[slot] = key;
        leaf->slot_data[slot] = value;
        leaf->num_slots_used++;

        if (splitnode && leaf != *splitnode && slot == leaf->num_slots_used - 1) {
            // special case: the node was split, and the insert is at the
            // last slot of the old node. then the splitkey must be
            // updated.
            *splitkey = key;
        }

        return true;
    }
}

// Split up a leaf node into two equally-filled sibling leaves. Returns
// the new nodes and it's insertion key in the two parameters.
static void btree_p_split_leaf(btree_p *const tree, btree_p_leaf_node *leaf, float *_newkey, void **_newleaf) {
    assert(btree_p_is_full(tree, leaf));

    unsigned int mid = (leaf->num_slots_used >> 1);


    btree_p_leaf_node *newleaf = btree_p_allocate_leaf();

    newleaf->num_slots_used = leaf->num_slots_used - mid;

    newleaf->next_leaf = leaf->next_leaf;
    if (newleaf->next_leaf == NULL) {
        assert(leaf == tree->last_leaf);
        tree->last_leaf = newleaf;
    } else {
        newleaf->next_leaf->prev_leaf = newleaf;
    }

    btree_p_copy_keys_fwd(leaf->slot_keys + mid, leaf->slot_keys + leaf->num_slots_used,
                          newleaf->slot_keys);
    btree_p_copy_data_fwd(leaf->slot_data + mid, leaf->slot_data + leaf->num_slots_used,
                          newleaf->slot_data);

    leaf->num_slots_used = mid;
    leaf->next_leaf = newleaf;
    newleaf->prev_leaf = leaf;

    *_newkey = leaf->slot_keys[leaf->num_slots_used - 1];
    *_newleaf = newleaf;
}

// Split up an inner node into two equally-filled sibling nodes. Returns
// the new nodes and it's insertion key in the two parameters. Requires
// the slot of the item will be inserted, so the nodes will be the same
// size after the insert.
static void btree_p_split_inner(btree_p *const tree, btree_p_inner_node *inner, float *_newkey, void **_newinner,
                                unsigned int addslot) {
    assert(btree_p_is_full(tree, inner));

    unsigned int mid = (inner->num_slots_used >> 1);


    // if the split is uneven and the overflowing item will be put into the
    // larger node, then the smaller split node may underflow
    if (addslot <= mid && mid > inner->num_slots_used - (mid + 1))
        mid--;

    btree_p_inner_node *newinner = btree_p_allocate_inner(inner->level);

    newinner->num_slots_used = inner->num_slots_used - (mid + 1);

    btree_p_copy_keys_fwd(inner->slot_keys + mid + 1, inner->slot_keys + inner->num_slots_used,
                          newinner->slot_keys);
    btree_p_copy_child_fwd(inner->slot_ptrs + mid + 1, inner->slot_ptrs + inner->num_slots_used + 1,
                           newinner->slot_ptrs);

    inner->num_slots_used = mid;

    *_newkey = inner->slot_keys[mid];
    *_newinner = newinner;
}


/* 
 * Bulk Loader - Construct Tree from Sorted Sequence
 */

// Save inner nodes and maxkey for next level.
typedef struct {
    btree_p_inner_node *first;
    const float *second;
} btree_p_next_level_info;

// Bulk load a sorted range. Loads items into leaves and constructs a
// B-tree above them. The tree must be empty when calling this function.
void btree_p_bulk_load(btree_p *const tree, float *keybegin, float *keyend, data_pt *databegin, data_pt *dataend) {
    assert(tree->root == NULL);
    size_t tmp1 = keyend - keybegin;
    size_t tmp2 = dataend - databegin;
    assert(tmp1 == tmp2);


    // calculate number of leaves needed, round up.
    size_t num_items = keyend - keybegin;
    size_t num_leaves = (num_items + tree->leaf_max_num_slots - 1) / tree->leaf_max_num_slots;
    tree->num_data = num_items;

    float *kt = keybegin;
    data_pt *dt = databegin;
    for (size_t i = 0; i < num_leaves; ++i) {
        // allocate new leaf node
        btree_p_leaf_node *leaf = btree_p_allocate_leaf();

        // copy keys or (key,value) pairs into leaf nodes, uses btree_p_set_slot().
        leaf->num_slots_used = (int) (num_items / (num_leaves - i));
        for (size_t s = 0; s < leaf->num_slots_used; ++s, ++kt, ++dt)
            btree_p_set_slot(leaf, s, *kt, *dt);

        if (tree->last_leaf != NULL) {
            tree->last_leaf->next_leaf = leaf;
            leaf->prev_leaf = tree->last_leaf;
        } else {
            tree->first_leaf = leaf;
        }
        tree->last_leaf = leaf;

        num_items -= leaf->num_slots_used;
    }

    assert(kt == keyend && dt == dataend && num_items == 0);

    // if the btree is so small to fit into one leaf, then we're done.
    if (tree->first_leaf == tree->last_leaf) {
        tree->root = tree->first_leaf;
        return;
    }


    // create first level of inner nodes, pointing to the leaves.
    size_t num_parents = (num_leaves + (tree->inner_max_num_slots + 1) - 1) / (tree->inner_max_num_slots + 1);


    btree_p_next_level_info *nextlevel = malloc(sizeof(btree_p_next_level_info) * num_parents);

    btree_p_leaf_node *leaf = tree->first_leaf;
    for (size_t i = 0; i < num_parents; ++i) {
        // allocate new inner node at level 1
        btree_p_inner_node *n = btree_p_allocate_inner(1);

        n->num_slots_used = (int) (num_leaves / (num_parents - i));
        assert(n->num_slots_used > 0);
        --n->num_slots_used; // this counts keys, but an inner node has keys+1 children.

        // copy last key from each leaf and set child
        for (unsigned short s = 0; s < n->num_slots_used; ++s) {
            n->slot_keys[s] = leaf->slot_keys[leaf->num_slots_used - 1];
            n->slot_ptrs[s] = leaf;
            leaf = leaf->next_leaf;
        }
        n->slot_ptrs[n->num_slots_used] = leaf;

        // track max key of any descendant.
        nextlevel[i].first = n;
        nextlevel[i].second = &leaf->slot_keys[leaf->num_slots_used - 1];

        leaf = leaf->next_leaf;
        num_leaves -= n->num_slots_used + 1;
    }

    assert(leaf == NULL && num_leaves == 0);

    // recursively build inner nodes pointing to inner nodes.
    for (int level = 2; num_parents != 1; ++level) {
        size_t num_children = num_parents;
        num_parents = (num_children + (tree->inner_max_num_slots + 1) - 1) / (tree->inner_max_num_slots + 1);

        size_t inner_index = 0;
        for (size_t i = 0; i < num_parents; ++i) {
            // allocate new inner node at level
            btree_p_inner_node *n = btree_p_allocate_inner(level);

            n->num_slots_used = (int) (num_children / (num_parents - i));
            assert(n->num_slots_used > 0);
            --n->num_slots_used; // this counts keys, but an inner node has keys+1 children.

            // copy children and maxkeys from nextlevel
            for (unsigned short s = 0; s < n->num_slots_used; ++s) {
                n->slot_keys[s] = *nextlevel[inner_index].second;
                n->slot_ptrs[s] = nextlevel[inner_index].first;
                ++inner_index;
            }
            n->slot_ptrs[n->num_slots_used] = nextlevel[inner_index].first;

            // reuse nextlevel array for parents, because we can overwrite
            // slots we've already consumed.
            nextlevel[i].first = n;
            nextlevel[i].second = nextlevel[inner_index].second;

            ++inner_index;
            num_children -= n->num_slots_used + 1;
        }

        assert(num_children == 0);
    }

    tree->root = nextlevel[0].first;
    //FREE nextlevel
    free(nextlevel);

}

// Constructor with a last_key value.
static inline btree_c_result btree_c_result_of(btree_c_result_flags f, const float k) {
    btree_c_result ans;
    ans.flags = f;
    ans.last_key = k;
    return ans;
}

// Test if this result object has a given flag set.
static inline bool btree_p_result_has(btree_c_result src, btree_c_result_flags f) {
    return (src.flags & f) != 0;
}

// To find the specific slot in a leaf node for deletion given key and value pair
// Node is considered a match if its key is within the range between key-(1e-4) and 
// key+(1e-4) inclusive and its ID is exactly equal to value. 
static int btree_p_delete_find_match(const btree_p_leaf_node *n, const float key, const long long value) {
    float keylow = key - 1e-4;
    float keyhi = key + 1e-4;
    int lo = 0;
    if (sizeof(n->slot_keys) > BTREE_BIN_SEARCH_THRESH) {
        if (n->num_slots_used == 0) return 0;

        int hi = n->num_slots_used;

        while (lo < hi) {
            int mid = (lo + hi) >> 1;

            if (keylow <= n->slot_keys[mid]) {
                hi = mid; // key-1e-4 <= mid
            } else {
                lo = mid + 1; // key-1e-4 > mid
            }
        }

    } else // for nodes <= BTREE_BIN_SEARCH_THRESH do linear search.
    {
        while (lo < n->num_slots_used && n->slot_keys[lo] < keylow) ++lo;
    }
    while (lo < n->num_slots_used && n->slot_keys[lo] <= keyhi) {
        if (n->slot_data[lo].info->id == value) return lo;
        lo++;
    }
    return -1;
}

// Merge two results OR-ing the result flags and overwriting last_keys.
static inline void btree_p_delete_merge_result(btree_c_result *dst, const btree_c_result other) {
    dst->flags = dst->flags | other.flags;

    // we overwrite existing last_keys on purpose
    if (btree_p_result_has(other, btree_update_last_key))
        dst->last_key = other.last_key;

    return;
}

// Delete the key/data pair.
// Node is considered a match if its key is within the range between key-(1e-4) and 
// key+(1e-4) inclusive and its ID is exactly equal to value. 
bool btree_p_delete(btree_p *const tree, const float key, const long long value) {

    if (!tree->root) return false;

    btree_c_result result = btree_p_delete_descend(tree, key, value, tree->root, NULL, NULL, NULL, NULL, NULL, 0);

    if (!btree_p_result_has(result, btree_not_found)) tree->num_data -= 1;
    return !btree_p_result_has(result, btree_not_found);
}

// Delete one key/data pair in the B+ tree. While descending down the tree, the
// id, left and right siblings and their parents are computed and
// passed down. The difficulty is that the iterator contains only a pointer
// to a btree_p_leaf_node, which means that this function must do a recursive depth
// first search for that leaf node in the subtree containing all pairs of
// the same key. This subtree can be very large, even the whole tree,
// though in practice it would not make sense to have so many duplicate
// keys. Once the referenced key/data pair is found, it is removed from the leaf.
// Node is considered a match if its key is within the range between key-(1e-4) and 
// key+(1e-4) inclusive and its ID is exactly equal to value. 
static btree_c_result
btree_p_delete_descend(btree_p *const tree, const float key, const long long value, void *curr, void *left,
                       void *right, btree_p_inner_node *leftparent, btree_p_inner_node *rightparent,
                       btree_p_inner_node *parent, unsigned int parentslot) {
    if (btree_p_is_leaf(curr)) {
        btree_p_leaf_node *leaf = (btree_p_leaf_node *) (curr);
        btree_p_leaf_node *leftleaf = (btree_p_leaf_node *) (left);
        btree_p_leaf_node *rightleaf = (btree_p_leaf_node *) (right);

        int slot = btree_p_delete_find_match(leaf, key, value);
        if (slot == -1) {
            return btree_c_result_of(btree_not_found, 0);
        }

        btree_p_copy_keys_fwd(leaf->slot_keys + slot + 1, leaf->slot_keys + leaf->num_slots_used,
                              leaf->slot_keys + slot);
        btree_p_copy_data_fwd(leaf->slot_data + slot + 1, leaf->slot_data + leaf->num_slots_used,
                              leaf->slot_data + slot);

        leaf->num_slots_used--;

        btree_c_result myres = btree_c_result_of(btree_ok, 0);

        // if the last key of the leaf was changed, the id is notified
        // and updates the key of this leaf
        if (slot == leaf->num_slots_used) {
            if (parent && parentslot < parent->num_slots_used) {
                assert(parent->slot_ptrs[parentslot] == curr);
                parent->slot_keys[parentslot] = leaf->slot_keys[leaf->num_slots_used - 1];
            } else {
                if (leaf->num_slots_used >= 1) {

                    btree_p_delete_merge_result(&myres, btree_c_result_of(btree_update_last_key,
                                                                          leaf->slot_keys[leaf->num_slots_used - 1]));
                } else {
                    assert(leaf == tree->root);
                }
            }
        }

        if (btree_p_is_below_min(tree, leaf) && !(leaf == tree->root && leaf->num_slots_used >= 1)) {
            // determine what to do about the underflow

            // case : if this empty leaf is the root, then delete all nodes
            // and set root to NULL.
            if (leftleaf == NULL && rightleaf == NULL) {
                assert(leaf == tree->root);
                assert(leaf->num_slots_used == 0);

                btree_p_free_node(tree->root);

                tree->root = leaf = NULL;
                tree->first_leaf = tree->last_leaf = NULL;

                return btree_c_result_of(btree_ok, 0);
            }
                // case : if both left and right leaves would underflow in case of
                // a shift, then merging is necessary. choose the more local merger
                // with our id
            else if ((leftleaf == NULL || btree_p_is_few(tree, leftleaf)) &&
                     (rightleaf == NULL || btree_p_is_few(tree, rightleaf))) {
                if (leftparent == parent)
                    btree_p_delete_merge_result(&myres, btree_p_merge_leaves(tree, leftleaf, leaf, leftparent));
                else
                    btree_p_delete_merge_result(&myres, btree_p_merge_leaves(tree, leaf, rightleaf, rightparent));
            }
                // case : the right leaf has extra data, so balance right with current
            else if ((leftleaf != NULL && btree_p_is_few(tree, leftleaf)) &&
                     (rightleaf != NULL && !btree_p_is_few(tree, rightleaf))) {
                if (rightparent == parent)
                    btree_p_delete_merge_result(&myres,
                                                btree_p_shift_left_leaf(leaf, rightleaf, rightparent, parentslot));
                else
                    btree_p_delete_merge_result(&myres, btree_p_merge_leaves(tree, leftleaf, leaf, leftparent));
            }
                // case : the left leaf has extra data, so balance left with current
            else if ((leftleaf != NULL && !btree_p_is_few(tree, leftleaf)) &&
                     (rightleaf != NULL && btree_p_is_few(tree, rightleaf))) {
                if (leftparent == parent)
                    btree_p_shift_right_leaf(leftleaf, leaf, leftparent, parentslot - 1);
                else
                    btree_p_delete_merge_result(&myres, btree_p_merge_leaves(tree, leaf, rightleaf, rightparent));
            }
                // case : both the leaf and right leaves have extra data and our
                // id, choose the leaf with more data
            else if (leftparent == rightparent) {
                if (leftleaf->num_slots_used <= rightleaf->num_slots_used)
                    btree_p_delete_merge_result(&myres,
                                                btree_p_shift_left_leaf(leaf, rightleaf, rightparent, parentslot));
                else
                    btree_p_shift_right_leaf(leftleaf, leaf, leftparent, parentslot - 1);
            } else {
                if (leftparent == parent)
                    btree_p_shift_right_leaf(leftleaf, leaf, leftparent, parentslot - 1);
                else
                    btree_p_delete_merge_result(&myres,
                                                btree_p_shift_left_leaf(leaf, rightleaf, rightparent, parentslot));
            }
        }

        return myres;
    } else // !curr->btree_p_is_leaf()
    {
        btree_p_inner_node *inner = (btree_p_inner_node *) (curr);
        btree_p_inner_node *leftinner = (btree_p_inner_node *) (left);
        btree_p_inner_node *rightinner = (btree_p_inner_node *) (right);

        // find first slot below which the searched iterator might be
        // located.

        btree_c_result result;
        int slot = btree_p_find_lower_inner(inner, key - 1e-4);

        while (slot <= inner->num_slots_used) {
            void *myleft, *myright;
            btree_p_inner_node *myleftparent, *myrightparent;

            if (slot == 0) {
                myleft = (left == NULL) ? NULL : ((btree_p_inner_node *) (left))->slot_ptrs[
                        ((btree_p_inner_node *) left)->num_slots_used - 1];
                myleftparent = leftparent;
            } else {
                myleft = inner->slot_ptrs[slot - 1];
                myleftparent = inner;
            }

            if (slot == inner->num_slots_used) {
                myright = (right == NULL) ? NULL : ((btree_p_inner_node *) (right))->slot_ptrs[0];
                myrightparent = rightparent;
            } else {
                myright = inner->slot_ptrs[slot + 1];
                myrightparent = inner;
            }


            result = btree_p_delete_descend(tree, key, value,
                                            inner->slot_ptrs[slot],
                                            myleft, myright,
                                            myleftparent, myrightparent,
                                            inner, slot);

            if (!btree_p_result_has(result, btree_not_found))
                break;

            // continue recursive search for leaf on next slot

            if (slot < inner->num_slots_used && inner->slot_keys[slot] > key + 1e-4)
                return btree_c_result_of(btree_not_found, 0);

            ++slot;
        }

        if (slot > inner->num_slots_used)
            return btree_c_result_of(btree_not_found, 0);

        btree_c_result myres = btree_c_result_of(btree_ok, 0);

        if (btree_p_result_has(result, btree_update_last_key)) {
            if (parent && parentslot < parent->num_slots_used) {
                assert(parent->slot_ptrs[parentslot] == curr);
                parent->slot_keys[parentslot] = result.last_key;
            } else {
                btree_p_delete_merge_result(&myres, btree_c_result_of(btree_update_last_key, result.last_key));
            }
        }

        if (btree_p_result_has(result, btree_fix_merge)) {
            // either the current node or the next is empty and should be removed
            if (btree_p_num_slots_used(inner->slot_ptrs[slot]) != 0)
                slot++;

            // this is the child slot invalidated by the merge
            assert(btree_p_num_slots_used(inner->slot_ptrs[slot]) == 0);

            btree_p_free_node(inner->slot_ptrs[slot]);

            btree_p_copy_keys_fwd(inner->slot_keys + slot, inner->slot_keys + inner->num_slots_used,
                                  inner->slot_keys + slot - 1);
            btree_p_copy_child_fwd(inner->slot_ptrs + slot + 1, inner->slot_ptrs + inner->num_slots_used + 1,
                                   inner->slot_ptrs + slot);

            inner->num_slots_used--;

            if (inner->level == 1) {
                // fix split key for children leaves
                slot--;
                btree_p_leaf_node *child = (btree_p_leaf_node *) (inner->slot_ptrs[slot]);
                inner->slot_keys[slot] = child->slot_keys[child->num_slots_used - 1];
            }
        }

        if (btree_p_is_below_min(tree, inner) && !(inner == tree->root && inner->num_slots_used >= 1)) {
            // case: the inner node is the root and has just one
            // child. that child becomes the new root
            if (leftinner == NULL && rightinner == NULL) {
                assert(inner == tree->root);
                assert(inner->num_slots_used == 0);

                tree->root = inner->slot_ptrs[0];

                inner->num_slots_used = 0;
                btree_p_free_node(inner);

                return btree_c_result_of(btree_ok, 0);
            }
                // case : if both left and right leaves would underflow in case of
                // a shift, then merging is necessary. choose the more local merger
                // with our id
            else if ((leftinner == NULL || btree_p_is_few(tree, leftinner)) &&
                     (rightinner == NULL || btree_p_is_few(tree, rightinner))) {
                if (leftparent == parent)
                    btree_p_delete_merge_result(&myres,
                                                btree_p_merge_inner(leftinner, inner, leftparent, parentslot - 1));
                else
                    btree_p_delete_merge_result(&myres,
                                                btree_p_merge_inner(inner, rightinner, rightparent, parentslot));
            }
                // case : the right leaf has extra data, so balance right with current
            else if ((leftinner != NULL && btree_p_is_few(tree, leftinner)) &&
                     (rightinner != NULL && !btree_p_is_few(tree, rightinner))) {
                if (rightparent == parent)
                    btree_p_shift_left_inner(inner, rightinner, rightparent, parentslot);
                else
                    btree_p_delete_merge_result(&myres,
                                                btree_p_merge_inner(leftinner, inner, leftparent, parentslot - 1));
            }
                // case : the left leaf has extra data, so balance left with current
            else if ((leftinner != NULL && !btree_p_is_few(tree, leftinner)) &&
                     (rightinner != NULL && btree_p_is_few(tree, rightinner))) {
                if (leftparent == parent)
                    btree_p_shift_right_inner(leftinner, inner, leftparent, parentslot - 1);
                else
                    btree_p_delete_merge_result(&myres,
                                                btree_p_merge_inner(inner, rightinner, rightparent, parentslot));
            }
                // case : both the leaf and right leaves have extra data and our
                // id, choose the leaf with more data
            else if (leftparent == rightparent) {
                if (leftinner->num_slots_used <= rightinner->num_slots_used)
                    btree_p_shift_left_inner(inner, rightinner, rightparent, parentslot);
                else
                    btree_p_shift_right_inner(leftinner, inner, leftparent, parentslot - 1);
            } else {
                if (leftparent == parent)
                    btree_p_shift_right_inner(leftinner, inner, leftparent, parentslot - 1);
                else
                    btree_p_shift_left_inner(inner, rightinner, rightparent, parentslot);
            }
        }

        return myres;
    }
}

// Merge two leaf nodes. The function moves all key/data pairs from right
// to left and sets right's num_slots_used to zero. The right slot is then
// removed by the calling id node.
static btree_c_result btree_p_merge_leaves(btree_p *const tree, btree_p_leaf_node *left, btree_p_leaf_node *right,
                                           btree_p_inner_node *parent) {
    (void) parent;

    assert(btree_p_is_leaf(left) && btree_p_is_leaf(right));
    assert(parent->level == 1);

    assert(left->num_slots_used + right->num_slots_used < tree->leaf_max_num_slots);

    btree_p_copy_keys_fwd(right->slot_keys, right->slot_keys + right->num_slots_used,
                          left->slot_keys + left->num_slots_used);
    btree_p_copy_data_fwd(right->slot_data, right->slot_data + right->num_slots_used,
                          left->slot_data + left->num_slots_used);

    left->num_slots_used += right->num_slots_used;

    left->next_leaf = right->next_leaf;
    if (left->next_leaf)
        left->next_leaf->prev_leaf = left;
    else
        tree->last_leaf = left;

    right->num_slots_used = 0;

    return btree_c_result_of(btree_fix_merge, 0);
}

// Merge two inner nodes. The function moves all key/slot_ptrs pairs from
// right to left and sets right's num_slots_used to zero. The right slot is then
// removed by the calling id node.
static btree_c_result
btree_p_merge_inner(btree_p_inner_node *left, btree_p_inner_node *right, btree_p_inner_node *parent,
                    unsigned int parentslot) {

    assert(left->level == right->level);
    assert(parent->level == left->level + 1);

    assert(parent->slot_ptrs[parentslot] == left);

    assert(left->num_slots_used + right->num_slots_used < BTREE_INNER_MAX_NUM_SLOTS);

    // retrieve the decision key from id
    left->slot_keys[left->num_slots_used] = parent->slot_keys[parentslot];
    left->num_slots_used++;

    // copy over keys and children from right
    btree_p_copy_keys_fwd(right->slot_keys, right->slot_keys + right->num_slots_used,
                          left->slot_keys + left->num_slots_used);
    btree_p_copy_child_fwd(right->slot_ptrs, right->slot_ptrs + right->num_slots_used + 1,
                           left->slot_ptrs + left->num_slots_used);

    left->num_slots_used += right->num_slots_used;
    right->num_slots_used = 0;

    return btree_c_result_of(btree_fix_merge, 0);
}

// Balance two leaf nodes. The function moves key/data pairs from right to
// left so that both nodes are equally filled. The id node is updated
// if possible.
static btree_c_result
btree_p_shift_left_leaf(btree_p_leaf_node *left, btree_p_leaf_node *right, btree_p_inner_node *parent,
                        unsigned int parentslot) {
    assert(btree_p_is_leaf(left) && btree_p_is_leaf(right));
    assert(parent->level == 1);

    assert(left->next_leaf == right);
    assert(left == right->prev_leaf);

    assert(left->num_slots_used < right->num_slots_used);
    assert(parent->slot_ptrs[parentslot] == left);

    unsigned int shiftnum = (right->num_slots_used - left->num_slots_used) >> 1;


    assert(left->num_slots_used + shiftnum < BTREE_LEAF_MAX_NUM_SLOTS);

    // copy the first items from the right node to the last slot in the left node.

    btree_p_copy_keys_fwd(right->slot_keys, right->slot_keys + shiftnum,
                          left->slot_keys + left->num_slots_used);
    btree_p_copy_data_fwd(right->slot_data, right->slot_data + shiftnum,
                          left->slot_data + left->num_slots_used);

    left->num_slots_used += shiftnum;

    // shift all slots in the right node to the left

    btree_p_copy_keys_fwd(right->slot_keys + shiftnum, right->slot_keys + right->num_slots_used,
                          right->slot_keys);
    btree_p_copy_data_fwd(right->slot_data + shiftnum, right->slot_data + right->num_slots_used,
                          right->slot_data);

    right->num_slots_used -= shiftnum;

    // fixup id
    if (parentslot < parent->num_slots_used) {
        parent->slot_keys[parentslot] = left->slot_keys[left->num_slots_used - 1];
        return btree_c_result_of(btree_ok, 0);
    } else { // the update is further up the tree
        return btree_c_result_of(btree_update_last_key, left->slot_keys[left->num_slots_used - 1]);
    }
}

// Balance two inner nodes. The function moves key/data pairs from right
// to left so that both nodes are equally filled. The id node is
// updated if possible.
static void btree_p_shift_left_inner(btree_p_inner_node *left, btree_p_inner_node *right, btree_p_inner_node *parent,
                                     unsigned int parentslot) {
    assert(left->level == right->level);
    assert(parent->level == left->level + 1);

    assert(left->num_slots_used < right->num_slots_used);
    assert(parent->slot_ptrs[parentslot] == left);

    unsigned int shiftnum = (right->num_slots_used - left->num_slots_used) >> 1;


    assert(left->num_slots_used + shiftnum < BTREE_INNER_MAX_NUM_SLOTS);

    // copy the id's decision slot_keys and slot_ptrs to the first new key on the left
    left->slot_keys[left->num_slots_used] = parent->slot_keys[parentslot];
    left->num_slots_used++;

    // copy the other items from the right node to the last slots in the left node.

    btree_p_copy_keys_fwd(right->slot_keys, right->slot_keys + shiftnum - 1,
                          left->slot_keys + left->num_slots_used);
    btree_p_copy_child_fwd(right->slot_ptrs, right->slot_ptrs + shiftnum,
                           left->slot_ptrs + left->num_slots_used);

    left->num_slots_used += shiftnum - 1;

    // fixup id
    parent->slot_keys[parentslot] = right->slot_keys[shiftnum - 1];

    // shift all slots in the right node

    btree_p_copy_keys_fwd(right->slot_keys + shiftnum, right->slot_keys + right->num_slots_used,
                          right->slot_keys);
    btree_p_copy_child_fwd(right->slot_ptrs + shiftnum, right->slot_ptrs + right->num_slots_used + 1,
                           right->slot_ptrs);

    right->num_slots_used -= shiftnum;
}

// Balance two leaf nodes. The function moves key/data pairs from left to
// right so that both nodes are equally filled. The id node is updated
// if possible.
static void btree_p_shift_right_leaf(btree_p_leaf_node *left, btree_p_leaf_node *right, btree_p_inner_node *parent,
                                     unsigned int parentslot) {
    assert(btree_p_is_leaf(left) && btree_p_is_leaf(right));
    assert(parent->level == 1);

    assert(left->next_leaf == right);
    assert(left == right->prev_leaf);
    assert(parent->slot_ptrs[parentslot] == left);

    assert(left->num_slots_used > right->num_slots_used);

    unsigned int shiftnum = (left->num_slots_used - right->num_slots_used) >> 1;


    // shift all slots in the right node

    assert(right->num_slots_used + shiftnum < BTREE_LEAF_MAX_NUM_SLOTS);

    btree_p_copy_keys_bwd(right->slot_keys, right->slot_keys + right->num_slots_used,
                          right->slot_keys + right->num_slots_used + shiftnum);
    btree_p_copy_data_bwd(right->slot_data, right->slot_data + right->num_slots_used,
                          right->slot_data + right->num_slots_used + shiftnum);

    right->num_slots_used += shiftnum;

    // copy the last items from the left node to the first slot in the right node.
    btree_p_copy_keys_fwd(left->slot_keys + left->num_slots_used - shiftnum, left->slot_keys + left->num_slots_used,
                          right->slot_keys);
    btree_p_copy_data_fwd(left->slot_data + left->num_slots_used - shiftnum, left->slot_data + left->num_slots_used,
                          right->slot_data);

    left->num_slots_used -= shiftnum;

    parent->slot_keys[parentslot] = left->slot_keys[left->num_slots_used - 1];
}

// Balance two inner nodes. The function moves key/data pairs from left to
// right so that both nodes are equally filled. The id node is updated
// if possible.
static void btree_p_shift_right_inner(btree_p_inner_node *left, btree_p_inner_node *right, btree_p_inner_node *parent,
                                      unsigned int parentslot) {
    assert(left->level == right->level);
    assert(parent->level == left->level + 1);

    assert(left->num_slots_used > right->num_slots_used);
    assert(parent->slot_ptrs[parentslot] == left);

    unsigned int shiftnum = (left->num_slots_used - right->num_slots_used) >> 1;


    // shift all slots in the right node

    assert(right->num_slots_used + shiftnum < BTREE_INNER_MAX_NUM_SLOTS);

    btree_p_copy_keys_bwd(right->slot_keys, right->slot_keys + right->num_slots_used,
                          right->slot_keys + right->num_slots_used + shiftnum);
    btree_p_copy_child_bwd(right->slot_ptrs, right->slot_ptrs + right->num_slots_used + 1,
                           right->slot_ptrs + right->num_slots_used + 1 + shiftnum);

    right->num_slots_used += shiftnum;

    // copy the id's decision slot_keys and slot_ptrs to the last new key on the right
    right->slot_keys[shiftnum - 1] = parent->slot_keys[parentslot];

    // copy the remaining last items from the left node to the first slot in the right node.
    btree_p_copy_keys_fwd(left->slot_keys + left->num_slots_used - shiftnum + 1, left->slot_keys + left->num_slots_used,
                          right->slot_keys);
    btree_p_copy_child_fwd(left->slot_ptrs + left->num_slots_used - shiftnum + 1,
                           left->slot_ptrs + left->num_slots_used + 1,
                           right->slot_ptrs);

    // copy the first to-be-removed key from the left node to the id's decision slot
    parent->slot_keys[parentslot] = left->slot_keys[left->num_slots_used - shiftnum];

    left->num_slots_used -= shiftnum;
}

// Tries to locate a key in the B+ tree and returns an iterator to the
// key/data slot if found. If unsuccessful it returns end().
btree_p_search_res btree_p_search(btree_p *const tree, const float key) {
    btree_p_search_res ret = {NULL, 0};
    void *n = tree->root;
    if (!n) return ret;

    while (!btree_p_is_leaf(n)) {
        const btree_p_inner_node *inner = (btree_p_inner_node *) (n);
        int slot = btree_p_find_lower_inner(inner, key);

        n = inner->slot_ptrs[slot];
    }

    btree_p_leaf_node *leaf = (btree_p_leaf_node *) (n);

    int slot = btree_p_find_lower_leaf(leaf, key);
    if (slot <= leaf->num_slots_used) {
        if (slot > 0) slot--;
        else {
            leaf = leaf->prev_leaf;
            if (!leaf) return ret;
            slot = leaf->num_slots_used - 1;
        }
        ret.n = leaf;
        ret.slot = slot;
        return ret;
    } else {
        // Should not reach here
        assert(0);
        return ret;
    }
}

bool btree_p_is_end(const btree_p *const tree, const btree_p_search_res src) {
    return src.n == NULL;
}

btree_p_search_res btree_p_find_prev(btree_p_search_res src) {
    assert(src.n != NULL);
    btree_p_search_res ret = {NULL, 0};
    btree_p_leaf_node *leaf = src.n;
    int slot = src.slot;
    if (slot > 0) slot--;
    else {
        leaf = leaf->prev_leaf;
        if (!leaf) return ret;
        slot = leaf->num_slots_used - 1;
    }
    ret.n = leaf;
    ret.slot = slot;
    return ret;
}

btree_p_search_res btree_p_find_next(btree_p_search_res src) {
    assert(src.n != NULL);
    btree_p_search_res ret = {NULL, 0};
    btree_p_leaf_node *leaf = src.n;
    int slot = src.slot;
    if (slot < leaf->num_slots_used - 1) slot++;
    else {
        leaf = leaf->next_leaf;
        if (!leaf) return ret;
        slot = 0;
    }
    ret.n = leaf;
    ret.slot = slot;
    return ret;
}

btree_p_search_res btree_p_first(const btree_p *const tree) {
    btree_p_search_res ret = {NULL, 0};
    ret.n = tree->first_leaf;
    ret.slot = 0;
    return ret;
}

btree_p_search_res btree_p_last(const btree_p *const tree) {
    btree_p_search_res ret = {NULL, 0};
    ret.n = tree->last_leaf;
    ret.slot = ret.n->num_slots_used - 1;
    return ret;
}

void btree_p_dump(btree_p *const tree) {
    void *root = tree->root;
    int l = 0, r = 0;
    void *q[100000];
    int depth[100000];
    q[0] = root;
    depth[0] = 0;
    int curr = 0;
    while (l <= r) {
        if (curr != depth[l]) {
            printf("\n\n\nline %d:\n", depth[l]);
            curr++;
        }
        if (btree_p_is_leaf(q[l])) {
            printf("[  ");
            btree_p_leaf_node *p = (btree_p_leaf_node *) q[l];
            for (int i = 0; i < p->num_slots_used; i++)
                printf("<%lf %lld> ", *(p->slot_keys + i), (p->slot_data + i)->info->id);
            printf(" ]  ");
        } else {
            btree_p_inner_node *p = (btree_p_inner_node *) q[l];
            printf("[  ");
            for (int i = 0; i < p->num_slots_used; i++) {
                printf("<%lf> ", *(p->slot_keys + i));
                r++;
                q[r] = *(p->slot_ptrs + i);
                depth[r] = depth[l] + 1;
            }
            r++;
            q[r] = *(p->slot_ptrs + p->num_slots_used);
            depth[r] = depth[l] + 1;
            printf(" ]  ");
        }
        l++;
    }
    printf("\n\n");
}

float btree_p_keyof(const btree_p_search_res src) {
    return src.n->slot_keys[src.slot];
}

data_pt btree_p_valueof(const btree_p_search_res src) {
    return src.n->slot_data[src.slot];
}
