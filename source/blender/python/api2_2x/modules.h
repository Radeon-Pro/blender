/* 
 *
 * ***** BEGIN GPL/BL DUAL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version. The Blender
 * Foundation also sells licenses for use in proprietary software under
 * the Blender License.  See http://www.blender.org/BL/ for information
 * about this.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * The Original Code is Copyright (C) 2001-2002 by NaN Holding BV.
 * All rights reserved.
 *
 * This is a new part of Blender.
 *
 * Contributor(s): Michel Selten
 *
 * ***** END GPL/BL DUAL LICENSE BLOCK *****
*/
#include <Python.h>

#include <DNA_object_types.h>
#include <DNA_camera_types.h>
#include <DNA_lamp_types.h>
#include <DNA_image_types.h>

/*****************************************************************************/
/* Global variables                                                          */
/*****************************************************************************/
extern PyObject *g_blenderdict;

void            M_Blender_Init (void);
PyObject *      M_Object_Init (void);
PyObject *      M_ObjectCreatePyObject (struct Object *obj);
int             M_ObjectCheckPyObject (PyObject *py_obj);
struct Object * M_ObjectFromPyObject (PyObject *py_obj);
PyObject *      M_NMesh_Init (void);
PyObject *      M_Camera_Init (void);
PyObject *      M_Lamp_Init (void);
PyObject *      M_Curve_Init (void);
PyObject *      M_Image_Init (void);
PyObject *      M_Window_Init (void);
PyObject *      M_Draw_Init (void);
PyObject *      M_BGL_Init (void);
PyObject *      M_Text_Init (void);
