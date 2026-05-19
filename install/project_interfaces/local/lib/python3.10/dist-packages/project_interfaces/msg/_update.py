# generated from rosidl_generator_py/resource/_idl.py.em
# with input from project_interfaces:msg/Update.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'ra'
# Member 'gamma_a'
# Member 'gamma_b'
# Member 'w1'
# Member 'w2'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Update(type):
    """Metaclass of message 'Update'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('project_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'project_interfaces.msg.Update')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__update
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__update
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__update
            cls._TYPE_SUPPORT = module.type_support_msg__msg__update
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__update

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Update(metaclass=Metaclass_Update):
    """Message class 'Update'."""

    __slots__ = [
        '_id_a',
        '_id_b',
        '_dim_a',
        '_dim_b',
        '_ra',
        '_gamma_a',
        '_gamma_b',
        '_w1',
        '_w2',
    ]

    _fields_and_field_types = {
        'id_a': 'int32',
        'id_b': 'int32',
        'dim_a': 'int32',
        'dim_b': 'int32',
        'ra': 'sequence<double>',
        'gamma_a': 'sequence<double>',
        'gamma_b': 'sequence<double>',
        'w1': 'sequence<double>',
        'w2': 'sequence<double>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.id_a = kwargs.get('id_a', int())
        self.id_b = kwargs.get('id_b', int())
        self.dim_a = kwargs.get('dim_a', int())
        self.dim_b = kwargs.get('dim_b', int())
        self.ra = array.array('d', kwargs.get('ra', []))
        self.gamma_a = array.array('d', kwargs.get('gamma_a', []))
        self.gamma_b = array.array('d', kwargs.get('gamma_b', []))
        self.w1 = array.array('d', kwargs.get('w1', []))
        self.w2 = array.array('d', kwargs.get('w2', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.id_a != other.id_a:
            return False
        if self.id_b != other.id_b:
            return False
        if self.dim_a != other.dim_a:
            return False
        if self.dim_b != other.dim_b:
            return False
        if self.ra != other.ra:
            return False
        if self.gamma_a != other.gamma_a:
            return False
        if self.gamma_b != other.gamma_b:
            return False
        if self.w1 != other.w1:
            return False
        if self.w2 != other.w2:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def id_a(self):
        """Message field 'id_a'."""
        return self._id_a

    @id_a.setter
    def id_a(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'id_a' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'id_a' field must be an integer in [-2147483648, 2147483647]"
        self._id_a = value

    @builtins.property
    def id_b(self):
        """Message field 'id_b'."""
        return self._id_b

    @id_b.setter
    def id_b(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'id_b' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'id_b' field must be an integer in [-2147483648, 2147483647]"
        self._id_b = value

    @builtins.property
    def dim_a(self):
        """Message field 'dim_a'."""
        return self._dim_a

    @dim_a.setter
    def dim_a(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'dim_a' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'dim_a' field must be an integer in [-2147483648, 2147483647]"
        self._dim_a = value

    @builtins.property
    def dim_b(self):
        """Message field 'dim_b'."""
        return self._dim_b

    @dim_b.setter
    def dim_b(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'dim_b' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'dim_b' field must be an integer in [-2147483648, 2147483647]"
        self._dim_b = value

    @builtins.property
    def ra(self):
        """Message field 'ra'."""
        return self._ra

    @ra.setter
    def ra(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'ra' array.array() must have the type code of 'd'"
            self._ra = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'ra' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._ra = array.array('d', value)

    @builtins.property
    def gamma_a(self):
        """Message field 'gamma_a'."""
        return self._gamma_a

    @gamma_a.setter
    def gamma_a(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'gamma_a' array.array() must have the type code of 'd'"
            self._gamma_a = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'gamma_a' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._gamma_a = array.array('d', value)

    @builtins.property
    def gamma_b(self):
        """Message field 'gamma_b'."""
        return self._gamma_b

    @gamma_b.setter
    def gamma_b(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'gamma_b' array.array() must have the type code of 'd'"
            self._gamma_b = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'gamma_b' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._gamma_b = array.array('d', value)

    @builtins.property
    def w1(self):
        """Message field 'w1'."""
        return self._w1

    @w1.setter
    def w1(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'w1' array.array() must have the type code of 'd'"
            self._w1 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'w1' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._w1 = array.array('d', value)

    @builtins.property
    def w2(self):
        """Message field 'w2'."""
        return self._w2

    @w2.setter
    def w2(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'w2' array.array() must have the type code of 'd'"
            self._w2 = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'w2' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._w2 = array.array('d', value)
