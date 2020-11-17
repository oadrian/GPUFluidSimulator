#include<GL/glut.h>
#include "FluidSim.h"
#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500

void display() {
	float const win_aspect = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, win_aspect, 1, 10);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Particle p(Vec3(0.f, 0.f, -5.f), 1.f);
	p.draw();

	glutSwapBuffers();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("SPH SIMULATION");
	glutDisplayFunc(display);
	glutMainLoop();
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
