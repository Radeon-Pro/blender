/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) Blender Foundation.
 * All rights reserved.
 *
 * The Original Code is: all of this file.
 *
 * Contributor(s): Lukas Toenne
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/blenvm/intern/bvm_api.cc
 *  \ingroup bvm
 */

#include "MEM_guardedalloc.h"

extern "C" {
#include "BLI_utildefines.h"
#include "BLI_listbase.h"

#include "DNA_node_types.h"

#include "BKE_effect.h"
#include "BKE_node.h"

#include "BVM_api.h"

#include "RNA_access.h"
}

#include "bvm_codegen.h"
#include "bvm_eval.h"
#include "bvm_expression.h"
#include "bvm_function.h"
#include "bvm_module.h"
#include "bvm_nodegraph.h"

void BVM_init(void)
{
	bvm::register_opcode_node_types();
}

void BVM_free(void)
{
}

/* ------------------------------------------------------------------------- */

BLI_INLINE bvm::Module *_MOD(struct BVMModule *mod)
{ return (bvm::Module *)mod; }

struct BVMModule *BVM_module_create(void)
{ return (struct BVMModule *)(new bvm::Module()); }

void BVM_module_free(BVMModule *mod)
{ delete _MOD(mod); }

struct BVMFunction *BVM_module_create_function(BVMModule *mod, const char *name)
{ return (struct BVMFunction *)_MOD(mod)->create_function(name); }

bool BVM_module_delete_function(BVMModule *mod, const char *name)
{ return _MOD(mod)->remove_function(name); }

/* ------------------------------------------------------------------------- */

BLI_INLINE bvm::Expression *_EXPR(struct BVMExpression *expr)
{ return (bvm::Expression *)expr; }

void BVM_expression_free(struct BVMExpression *expr)
{ delete _EXPR(expr); }

/* ------------------------------------------------------------------------- */

BLI_INLINE bvm::NodeGraph *_GRAPH(struct BVMNodeGraph *graph)
{ return (bvm::NodeGraph *)graph; }
BLI_INLINE bvm::NodeInstance *_NODE(struct BVMNodeInstance *node)
{ return (bvm::NodeInstance *)node; }

struct BVMNodeInstance *BVM_nodegraph_add_node(BVMNodeGraph *graph, const char *type, const char *name)
{ return (struct BVMNodeInstance *)_GRAPH(graph)->add_node(type, name); }

/* ------------------------------------------------------------------------- */

BLI_INLINE bvm::EvalContext *_CTX(struct BVMEvalContext *ctx)
{ return (bvm::EvalContext *)ctx; }

struct BVMEvalContext *BVM_context_create(void)
{ return (BVMEvalContext *)(new bvm::EvalContext()); }

void BVM_context_free(struct BVMEvalContext *ctx)
{ delete _CTX(ctx); }

void BVM_eval_forcefield(struct BVMEvalContext *ctx, struct BVMExpression *expr,
                         const EffectedPoint *point, float force[3], float impulse[3])
{
	bvm::EvalData data;
	data.effector.position = bvm::float3(point->loc[0], point->loc[1], point->loc[2]);
	data.effector.velocity = bvm::float3(point->vel[0], point->vel[1], point->vel[2]);
	void *results[] = { force, impulse };
	
	_CTX(ctx)->eval_expression(&data, *_EXPR(expr), results);
}

/* ------------------------------------------------------------------------- */

typedef std::pair<bNode*, bNodeSocket*> bSocketPair;
typedef std::pair<bvm::NodeInstance*, bvm::string> SocketPair;
typedef std::map<bSocketPair, SocketPair> SocketMap;

static void map_input_socket(SocketMap &socket_map, bNode *bnode, int bindex, bvm::NodeInstance *node, const bvm::string &name)
{
	bNodeSocket *binput = (bNodeSocket *)BLI_findlink(&bnode->inputs, bindex);
	
	socket_map[bSocketPair(bnode, binput)] = SocketPair(node, name);
	
	switch (binput->type) {
		case SOCK_FLOAT: {
			bNodeSocketValueFloat *bvalue = (bNodeSocketValueFloat *)binput->default_value;
			node->set_input_value(name, bvalue->value);
			break;
		}
		case SOCK_VECTOR: {
			bNodeSocketValueVector *bvalue = (bNodeSocketValueVector *)binput->default_value;
			node->set_input_value(name, bvm::float3(bvalue->value[0], bvalue->value[1], bvalue->value[2]));
			break;
		}
	}
}

static void map_output_socket(SocketMap &socket_map, bNode *bnode, int bindex, bvm::NodeInstance *node, const bvm::string &name)
{
	bNodeSocket *boutput = (bNodeSocket *)BLI_findlink(&bnode->outputs, bindex);
	
	socket_map[bSocketPair(bnode, boutput)] = SocketPair(node, name);
}

static void map_all_sockets(SocketMap &socket_map, bNode *bnode, bvm::NodeInstance *node)
{
	bNodeSocket *bsock;
	int i;
	for (bsock = (bNodeSocket *)bnode->inputs.first, i = 0; bsock; bsock = bsock->next, ++i) {
		const bvm::NodeSocket *input = node->type->find_input(i);
		map_input_socket(socket_map, bnode, i, node, input->name);
	}
	for (bsock = (bNodeSocket *)bnode->outputs.first, i = 0; bsock; bsock = bsock->next, ++i) {
		const bvm::NodeSocket *output = node->type->find_output(i);
		map_output_socket(socket_map, bnode, i, node, output->name);
	}
}

static void binary_math_node(bvm::NodeGraph &graph, SocketMap &socket_map, bNode *bnode, const bvm::string &type)
{
	bvm::NodeInstance *node = graph.add_node(type, bnode->name);
	map_input_socket(socket_map, bnode, 0, node, "value_a");
	map_input_socket(socket_map, bnode, 1, node, "value_b");
	map_output_socket(socket_map, bnode, 0, node, "value");
}

static void unary_math_node(bvm::NodeGraph &graph, SocketMap &socket_map, bNode *bnode, const bvm::string &type)
{
	bvm::NodeInstance *node = graph.add_node(type, bnode->name);
	bNodeSocket *sock0 = (bNodeSocket *)BLI_findlink(&bnode->inputs, 0);
	bNodeSocket *sock1 = (bNodeSocket *)BLI_findlink(&bnode->inputs, 1);
	bool sock0_linked = !nodeSocketIsHidden(sock0) && (sock0->flag & SOCK_IN_USE);
	bool sock1_linked = !nodeSocketIsHidden(sock1) && (sock1->flag & SOCK_IN_USE);
	if (sock0_linked || !sock1_linked)
		map_input_socket(socket_map, bnode, 0, node, "value");
	else
		map_input_socket(socket_map, bnode, 1, node, "value");
	map_output_socket(socket_map, bnode, 0, node, "value");
}

static void gen_forcefield_nodegraph(bNodeTree *btree, bvm::NodeGraph &graph)
{
	{
		float zero[3] = {0.0f, 0.0f, 0.0f};
		graph.add_output("force", BVM_FLOAT3, zero);
		graph.add_output("impulse", BVM_FLOAT3, zero);
	}
	
	/* maps bNodeTree sockets to internal sockets, for converting links */
	SocketMap socket_map;
	
#if 1
	for (bNode *bnode = (bNode*)btree->nodes.first; bnode; bnode = bnode->next) {
		PointerRNA ptr;
		RNA_pointer_create((ID *)btree, &RNA_Node, bnode, &ptr);
		
		BLI_assert(bnode->typeinfo != NULL);
		if (!nodeIsRegistered(bnode))
			continue;
		
		const char *type = bnode->typeinfo->idname;
#if 0
		/*NodeInstance *node =*/ graph.add_node(type, bnode->name);
#else
		if (bvm::string(type) == "ForceOutputNode") {
			{
				bvm::NodeInstance *node = graph.add_node("PASS_FLOAT3", "RET_FORCE_" + bvm::string(bnode->name));
				map_input_socket(socket_map, bnode, 0, node, "value");
				map_output_socket(socket_map, bnode, 0, node, "value");
				
				graph.set_output_link("force", node, "value");
			}
			
			{
				bvm::NodeInstance *node = graph.add_node("PASS_FLOAT3", "RET_IMPULSE_" + bvm::string(bnode->name));
				map_input_socket(socket_map, bnode, 1, node, "value");
				map_output_socket(socket_map, bnode, 0, node, "value");
				
				graph.set_output_link("impulse", node, "value");
			}
		}
		else if (bvm::string(type) == "ObjectSeparateVectorNode") {
			{
				bvm::NodeInstance *node = graph.add_node("GET_ELEM0_FLOAT3", "GET_ELEM0_" + bvm::string(bnode->name));
				map_input_socket(socket_map, bnode, 0, node, "value");
				map_output_socket(socket_map, bnode, 0, node, "value");
			}
			{
				bvm::NodeInstance *node = graph.add_node("GET_ELEM1_FLOAT3", "GET_ELEM1_" + bvm::string(bnode->name));
				map_input_socket(socket_map, bnode, 0, node, "value");
				map_output_socket(socket_map, bnode, 1, node, "value");
			}
			{
				bvm::NodeInstance *node = graph.add_node("GET_ELEM2_FLOAT3", "GET_ELEM2_" + bvm::string(bnode->name));
				map_input_socket(socket_map, bnode, 0, node, "value");
				map_output_socket(socket_map, bnode, 2, node, "value");
			}
		}
		else if (bvm::string(type) == "ObjectCombineVectorNode") {
			bvm::NodeInstance *node = graph.add_node("SET_FLOAT3", bvm::string(bnode->name));
			map_input_socket(socket_map, bnode, 0, node, "value_x");
			map_input_socket(socket_map, bnode, 1, node, "value_y");
			map_input_socket(socket_map, bnode, 2, node, "value_z");
			map_output_socket(socket_map, bnode, 0, node, "value");
		}
		else if (bvm::string(type) == "ForceEffectorDataNode") {
			{
				bvm::NodeInstance *node = graph.add_node("EFFECTOR_POSITION", "EFFECTOR_POS" + bvm::string(bnode->name));
				map_output_socket(socket_map, bnode, 0, node, "value");
			}
			{
				bvm::NodeInstance *node = graph.add_node("EFFECTOR_VELOCITY", "EFFECTOR_VEL" + bvm::string(bnode->name));
				map_output_socket(socket_map, bnode, 1, node, "value");
			}
		}
		else if (bvm::string(type) == "ObjectMathNode") {
			int mode = RNA_enum_get(&ptr, "mode");
			switch (mode) {
				case 0: binary_math_node(graph, socket_map, bnode, "ADD_FLOAT"); break;
				case 1: binary_math_node(graph, socket_map, bnode, "SUB_FLOAT"); break;
				case 2: binary_math_node(graph, socket_map, bnode, "MUL_FLOAT"); break;
				case 3: binary_math_node(graph, socket_map, bnode, "DIV_FLOAT"); break;
				case 4: unary_math_node(graph, socket_map, bnode, "SINE"); break;
				case 5: unary_math_node(graph, socket_map, bnode, "COSINE"); break;
				case 6: unary_math_node(graph, socket_map, bnode, "TANGENT"); break;
				case 7: unary_math_node(graph, socket_map, bnode, "ARCSINE"); break;
				case 8: unary_math_node(graph, socket_map, bnode, "ARCCOSINE"); break;
				case 9: unary_math_node(graph, socket_map, bnode, "ARCTANGENT"); break;
				case 10: binary_math_node(graph, socket_map, bnode, "POWER"); break;
				case 11: binary_math_node(graph, socket_map, bnode, "LOGARITHM"); break;
				case 12: binary_math_node(graph, socket_map, bnode, "MINIMUM"); break;
				case 13: binary_math_node(graph, socket_map, bnode, "MAXIMUM"); break;
				case 14: unary_math_node(graph, socket_map, bnode, "ROUND"); break;
				case 15: binary_math_node(graph, socket_map, bnode, "LESS_THAN"); break;
				case 16: binary_math_node(graph, socket_map, bnode, "GREATER_THAN"); break;
				case 17: binary_math_node(graph, socket_map, bnode, "MODULO"); break;
				case 18: unary_math_node(graph, socket_map, bnode, "ABSOLUTE"); break;
				case 19: unary_math_node(graph, socket_map, bnode, "CLAMP"); break;
			}
		}
		else if (bvm::string(type) == "ObjectVectorMathNode") {
			int mode = RNA_enum_get(&ptr, "mode");
			switch (mode) {
				case 0: {
					bvm::NodeInstance *node = graph.add_node("ADD_FLOAT3", bnode->name);
					map_input_socket(socket_map, bnode, 0, node, "value_a");
					map_input_socket(socket_map, bnode, 1, node, "value_b");
					map_output_socket(socket_map, bnode, 0, node, "value");
					break;
				}
				case 1: {
					bvm::NodeInstance *node = graph.add_node("SUB_FLOAT3", bnode->name);
					map_input_socket(socket_map, bnode, 0, node, "value_a");
					map_input_socket(socket_map, bnode, 1, node, "value_b");
					map_output_socket(socket_map, bnode, 0, node, "value");
					break;
				}
			}
		}
#endif
	}
	
	for (bNodeLink *blink = (bNodeLink *)btree->links.first; blink; blink = blink->next) {
		if (!(blink->flag & NODE_LINK_VALID))
			continue;
		
		SocketMap::const_iterator it_from = socket_map.find(bSocketPair(blink->fromnode, blink->fromsock));
		SocketMap::const_iterator it_to = socket_map.find(bSocketPair(blink->tonode, blink->tosock));
		if (it_from != socket_map.end() && it_to != socket_map.end()) {
			SocketPair from_pair = it_from->second;
			SocketPair to_pair = it_to->second;
			graph.add_link(from_pair.first, from_pair.second,
			               to_pair.first, to_pair.second);
		}
	}
#else
	// XXX TESTING
	{
		bvm::NodeInstance *node = graph.add_node("PASS_FLOAT3", "pass0");
		node->set_input_value("value", bvm::float3(0.5, 1.5, -0.3));
		
		graph.set_output_link("force", node, "value");
	}
#endif
}

struct BVMExpression *BVM_gen_forcefield_expression(bNodeTree *btree)
{
	using namespace bvm;
	
	NodeGraph graph;
	gen_forcefield_nodegraph(btree, graph);
	
	BVMCompiler compiler;
	Expression *expr = compiler.codegen_expression(graph);
	
	return (BVMExpression *)expr;
}

#undef _GRAPH

#undef _EXPR
