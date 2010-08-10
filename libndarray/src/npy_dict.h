/*--------------------------------------------------------------------------*\
 *                   -----===== HashTable =====-----
 *
 * Author: Keith Pomakis (pomakis@pobox.com)
 * Date:   August, 1998
 * Released to the public domain.
 *
 *--------------------------------------------------------------------------
 * $Id: hashtable.c,v 1.4 2000/08/02 19:01:13 pomakis Exp pomakis $
 \*--------------------------------------------------------------------------*/

/* Adapted for use in NumPy to replace usage of CPython PyDict functions.
 The bulk of the edits are for NumPy style guidelines and function naming. */

#ifndef _NPY_DICT_H_
#define _NPY_DICT_H_

struct NpyDict_KVPair_struct;
struct NpyDict_struct;

typedef struct NpyDict_KVPair_struct NpyDict_KVPair;
typedef struct NpyDict_struct NpyDict;

struct NpyDict_Iter {
    long bucket;
    NpyDict_KVPair *element;
};

typedef struct NpyDict_Iter NpyDict_Iter;




/*
 * Dictionary APIs
 */
NpyDict *NpyDict_CreateTable(long numOfBuckets);
void NpyDict_Destroy(NpyDict *hashTable);
NpyDict *NpyDict_Copy(const NpyDict *orig, void *(*copyKey)(void *), void *(*copyValue)(void *));
int NpyDict_ContainsKey(const NpyDict *hashTable, const void *key);
int NpyDict_ContainsValue(const NpyDict *hashTable, const void *value);
int NpyDict_Put(NpyDict *hashTable, const void *key, void *value);
void NpyDict_ForceValue(NpyDict *hashTable, const void *key, void *newValue);
void *NpyDict_Get(const NpyDict *hashTable, const void *key);
void NpyDict_Rekey(NpyDict *hashTable, const void *oldKey, const void *newKey);
void NpyDict_Remove(NpyDict *hashTable, const void *key);
void NpyDict_RemoveAll(NpyDict *hashTable);
void NpyDict_IterInit(NpyDict_Iter *iter);
int NpyDict_IterNext(NpyDict *hashTable, NpyDict_Iter *iter, void **key, void **value);
int NpyDict_IsEmpty(const NpyDict *hashTable);
long NpyDict_Size(const NpyDict *hashTable);
long NpyDict_GetNumBuckets(const NpyDict *hashTable);
void NpyDict_SetKeyComparisonFunction(NpyDict *hashTable,
                                      int (*keycmp)(const void *key1, const void *key2));
void NpyDict_SetValueComparisonFunction(NpyDict *hashTable,
                                        int (*valuecmp)(const void *value1, const void *value2));
void NpyDict_SetHashFunction(NpyDict *hashTable,
                             unsigned long (*hashFunction)(const void *key));
void NpyDict_Rehash(NpyDict *hashTable, long numOfBuckets);
void NpyDict_SetIdealRatio(NpyDict *hashTable, float idealRatio,
                           float lowerRehashThreshold, float upperRehashThreshold);
void NpyDict_SetDeallocationFunctions(NpyDict *hashTable,
                                      void (*keyDeallocator)(void *key),
                                      void (*valueDeallocator)(void *value));
unsigned long NpyDict_StringHashFunction(const void *key);

#endif

