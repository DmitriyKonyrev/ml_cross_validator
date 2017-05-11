#ifndef VP_TREE_H
#define VP_TREE_H
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <stdio.h>
#include <queue>
#include <limits>

namespace DataStructures
{

	template<typename T>
	class VpTree
	{
	public:
	   	typedef std::function<double( const T&, const T&)> dist_type;

	public:
		VpTree() : _root(0) {}
		VpTree(const dist_type& dist) : distance(dist), _root(0) {}

		~VpTree() {
		    delete _root;
		}

		size_t size() const
		{
			return _items.size();
		}

		void create( const std::vector<T>& items ) {
		    delete _root;
		    _items = items;
		    _root = buildFromPoints(0, items.size());
		}

		void search( const T& target, int k, std::vector<T>* results, 
		    std::vector<double>* distances) 
		{
		    std::priority_queue<HeapItem> heap;

		    _tau = std::numeric_limits<double>::max();
		    search( _root, target, k, heap );

		    results->clear(); distances->clear();

		    while( !heap.empty() ) {
		        results->push_back( _items[heap.top().index] );
		        distances->push_back( heap.top().dist );
		        heap.pop();
		    }

		    std::reverse( results->begin(), results->end() );
		    std::reverse( distances->begin(), distances->end() );
		}

	private:
		std::vector<T> _items;
		double _tau;
		dist_type distance;

		struct Node 
		{
		    int index;
		    double threshold;
		    Node* left;
		    Node* right;

		    Node() :
		        index(0), threshold(0.), left(0), right(0) {}

		    ~Node() {
		        delete left;
		        delete right;
		    }
		}* _root;

		struct HeapItem {
		    HeapItem( int index, double dist) :
		        index(index), dist(dist) {}
		    int index;
		    double dist;
		    bool operator<( const HeapItem& o ) const {
		        return dist < o.dist;   
		    }
		};

		struct DistanceComparator
		{
		    const T& item;
			const dist_type& distance;
		    DistanceComparator( const T& item, const dist_type& dist ) 
			: item(item)
			, distance(dist)
			{}
		    bool operator()(const T& a, const T& b) {
		        return distance( item, a ) < distance( item, b );
		    }
		};

		Node* buildFromPoints( int lower, int upper )
		{
		    if ( upper == lower ) {
		        return NULL;
		    }

		    Node* node = new Node();
		    node->index = lower;

		    if ( upper - lower > 1 ) {

		        // choose an arbitrary point and move it to the start
		        int i = (int)((double)rand() / RAND_MAX * (upper - lower - 1) ) + lower;
		        std::swap( _items[lower], _items[i] );

		        int median = ( upper + lower ) / 2;

		        // partitian around the median distance
		        std::nth_element( 
		            _items.begin() + lower + 1, 
		            _items.begin() + median,
		            _items.begin() + upper,
		            DistanceComparator( _items[lower], distance ));

		        // what was the median?
		        node->threshold = distance( _items[lower], _items[median] );

		        node->index = lower;
		        node->left = buildFromPoints( lower + 1, median );
		        node->right = buildFromPoints( median, upper );
		    }

		    return node;
		}

		void search( Node* node, const T& target, int k,
		             std::priority_queue<HeapItem>& heap )
		{
		    if ( node == NULL ) return;

		    double dist = distance( _items[node->index], target );
		    //printf("dist=%g tau=%gn", dist, _tau );

		    if ( dist < _tau ) {
		        if ( heap.size() == k ) heap.pop();
		        heap.push( HeapItem(node->index, dist) );
		        if ( heap.size() == k ) _tau = heap.top().dist;
		    }

		    if ( node->left == NULL && node->right == NULL ) {
		        return;
		    }

		    if ( dist < node->threshold ) {
		        if ( dist - _tau <= node->threshold ) {
		            search( node->left, target, k, heap );
		        }

		        if ( dist + _tau >= node->threshold ) {
		            search( node->right, target, k, heap );
		        }

		    } else {
		        if ( dist + _tau >= node->threshold ) {
		            search( node->right, target, k, heap );
		        }

		        if ( dist - _tau <= node->threshold ) {
		            search( node->left, target, k, heap );
		        }
		    }
		}
	};
}
#endif //VP_TREE_H
