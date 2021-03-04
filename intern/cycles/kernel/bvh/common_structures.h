//#define OBJECT_NONE 0xffffffff
//#define PRIM_NONE   0xffffffff

#define TRIANGLE0 0
#define TRIANGLE1 1
#define TRIANGLE2 2
#define TRIANGLE3 3
#define BOX16     4
#define BOX32     5
#define OBJECT    6
#define USER      7



#define CL_AMD_TRAVERSAL_STATUS_UNKNOWN 0
#define CL_AMD_TRAVERSAL_STATUS_HIT 1
#define CL_AMD_TRAVERSAL_STATUS_FINISHED 2

#define CL_AMD_HIT_TYPE_TRIANGLE_NODE 1
#define CL_AMD_HIT_TYPE_USER_NODE 2

#define CL_AMD_TRAVERSAL_FLAGS_ACCEPT_FIRST_HIT 1
#define CL_AMD_TRAVERSAL_FLAGS_ACCEPT_CLOSEST_HIT 2 

struct hit_t;
struct traversal_t;

typedef uint traversal_status_t;
typedef uint traversal_flags_t;

/** Ray structure **/
typedef struct __attribute__((packed)) ray_t
{
    float3 origin;
    float  t_min;
    float3 direction;
    float  t_max;
};

#define TOP_LEVEL 0x80000000
#define GET_NODE(node) ((node & ~TOP_LEVEL) >> 3u)
#define NODE_TYPE(A)   ((A)&7u)
#define INVALID_NODE   0xffffffff

typedef struct Node
{
    uint data[16];
} Node;

typedef struct TriangleNode
{
    float v0x;
    float v0y;
    float v0z;
    
    float v1x;
    float v1y;
    float v1z;
    
    float v2x;
    float v2y;
    float v2z;
    
    float v3x;
    float v3y;
    float v3z;
    
    //float v4x;
    uint triangle_id;
    float v4y;
    float v4z;

    uint id;  // triangle_id;
} TriangleNode;

typedef struct BoxNodeF32
{
    uint child0;
    uint child1;
    uint child2;
    uint child3;
    
    float box0_min_x;
    float box0_min_y;
    float box0_min_z;
    float box0_max_x;
    float box0_max_y;
    float box0_max_z;
    
    float box1_min_x;
    float box1_min_y;
    float box1_min_z;
    float box1_max_x;
    float box1_max_y;
    float box1_max_z;
    
    float box2_min_x;
    float box2_min_y;
    float box2_min_z;
    float box2_max_x;
    float box2_max_y;
    float box2_max_z;
    
    float box3_min_x;
    float box3_min_y;
    float box3_min_z;
    float box3_max_x;
    float box3_max_y;
    float box3_max_z;
    
    uint padding0;
    uint padding1;
    uint padding2;
    uint padding3;
} BoxNodeF32;


typedef struct CustomNode
{
    uint size;
    uint data[124];
} CustomNode;
/*
typedef struct Transform
{
    float4 x, y, z;
} Transform;*/

typedef struct ObjectNode
{
    Transform tfm;
    uint shape_id;
    uint bottom_pointer;
    uint prim_offset;
    uint padding1;
} ObjectNode;

typedef struct hit_t
{
    traversal_status_t status;
    uint instance_id;
    uint primitive_id;
    float t;
    float2 uv;
} hit_t;

typedef struct traversal_t
{
    __global char* node_ptr;
    struct ray_t ray;
    struct hit_t hit;
    traversal_flags_t flag;
} traversal_t;
