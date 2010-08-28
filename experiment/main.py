import interface


print interface.create_new()


import numpy

a = numpy.array([[3,2,5], [1,4,6]])

interface.receive_array(3.14)
interface.receive_array(42)
interface.receive_array([[8,2,5], [7,4,6]])
interface.receive_array((1.3, 5.6, 7.8))
interface.receive_array(a)
interface.receive_array("foo")

print interface.create_new2()
