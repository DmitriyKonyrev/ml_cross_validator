#ifndef MATHMATRIX_H
#define MATHMATRIX_H

#include <iostream>
#include <vector>
#include "mathvector.h"

#include "mathmatrix_invertor.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MathCore
{
	namespace AlgebraCore
	{
		namespace MatrixCore
		{
			namespace MatrixAlgorithm
			{
				namespace MatrixDecomposer
				{
					template <typename T> class MathMatrixDecomposer;
				}

				namespace MatrixSolver
				{
					template <typename T> class MathMatrixSolver;
				}


			};

			template<typename T> class MathMatrix
			{
			protected:

				size_t row_size;
				size_t col_size;

				std::vector<MathVector<T>> values;

			public:

				//---------------------------friends-------------------------------------
				friend MatrixAlgorithm::MatrixInvertor::MathMatrixInvertor<T>;
				friend MatrixAlgorithm::MatrixDecomposer::MathMatrixDecomposer<T>;
				friend MatrixAlgorithm::MatrixSolver::MathMatrixSolver<T>;
				//-----------------------------------------------------------------------

			protected:

				MatrixAlgorithm::MatrixInvertor::MathMatrixInvertor< T >* invertor;

			public:
				//---------------------------constructors---------------------------------
				MathMatrix();
				MathMatrix(size_t _rows_size, size_t _cols_size, T _default_value = 0);
				MathMatrix(const MathMatrix& x);
				MathMatrix(std::vector<std::vector<T>>& _matrix);
				MathMatrix(std::vector<MathVector<T>>& _math_matrix);
				MathMatrix(size_t size, T _default_value = 0);

				void setValues(std::vector<MathVector<T>>& _math_matrix);
				static MathMatrix& createIdentityMatrix(size_t size, T _default_value = 0);
				//------------------------------------------------------------------------

				//--------------------------usual-operators------------------------------
				size_t rows_size() const;
				size_t cols_size() const;

				MathMatrix& clear();

				void push_back(MathVector<T> row);
				void pop_back();

				void pop_at(size_t position);

				void set_invertor(MatrixAlgorithm::MatrixInvertor::MathMatrixInvertor<T> invertor);

				void insert_element(T elemenet, size_t rowIndex, size_t colIndex);
                MathVector<T> row(size_t rowIndex);
				T at(size_t rowIndex, size_t colIndex);
				//---------------------------------------------------------------------

				//-------------------------operators------------------------------------
				MathMatrix& operator=(MathMatrix const & _MathMatrix);

				//MathVector<T>& operator*(MathVector<T> &_vector);

				MathMatrix& operator+(MathMatrix &_MathMatrix);
				MathMatrix& operator-(MathMatrix &_MathMatrix);
				MathMatrix& operator*(MathMatrix &_MathMatrix);
				MathMatrix& raw_multiply(MathMatrix &_MathMatrix);

				MathMatrix& operator+=(MathMatrix &_MathMatrix);
				MathMatrix& operator-=(MathMatrix &_MathMatrix);
				MathMatrix& operator*=(MathMatrix &_MathMatrix);

				MathMatrix& operator+(const T& _value);
				MathMatrix& operator-(const T& _value);
				MathMatrix& operator*(const T& _value);
				MathMatrix& operator/(const T& _value);

				MathMatrix& operator+=(const T& _value);
				MathMatrix& operator-=(const T& _value);
				MathMatrix& operator*=(const T& _value);
				MathMatrix& operator/=(const T& _value);

				MathMatrix& operator!();
				MathMatrix& operator~();

				bool operator==(MathMatrix &_MathMatrix) const;
				bool operator!=(MathMatrix &_MathMatrix) const;

				MathVector<T>& operator[](size_t _row_index);
				//----------------------------------------------------------------------
			};

			//---------------------------constructors---------------------------------
			template<typename T> MathMatrix<T>::MathMatrix()
				: row_size(0), col_size(0)
			{
				this->invertor = new MatrixAlgorithm::MatrixInvertor::QRInvertor<T>();
			}

			template<typename T> MathMatrix<T>::MathMatrix(size_t _rows_size, size_t _cols_size, T _default_value)
				: row_size(_rows_size), col_size(_cols_size)
			{
				this->values.resize(this->row_size, MathVector<T>(col_size, _default_value));
				this->invertor = new MatrixAlgorithm::MatrixInvertor::QRInvertor<T>();
			}

			template<typename T> MathMatrix<T>::MathMatrix(size_t _size, T _default_value)
				: row_size(_size), col_size(_size)
			{
				this->values.resize(this->row_size, MathVector<T>(col_size, 0));
				this->invertor = new MatrixAlgorithm::MatrixInvertor::QRInvertor<T>();

				for (size_t rowIndex = 0; rowIndex < _size; ++rowIndex)
				{
					this->values.at(rowIndex).insert(_default_value, rowIndex);
				}
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::createIdentityMatrix(size_t size, T _default_value)
			{
				std::vector<MathVector<T>> identityValues(size, MathVector<T>(size, 0.));

				for (size_t index = 0; index < size; ++index)
				{
					identityValues.at(index).insert(_default_value, index);
				}

				MathMatrix<T>* identityMatrix = new MathMatrix<T>(identityValues);

				return *identityMatrix;
			}

			template<typename T> MathMatrix<T>::MathMatrix(const MathMatrix& x)
			{
				this->row_size = x.row_size;
				this->col_size = x.col_size;

				this->values = x.values;
				this->invertor = x.invertor;
			}

			template<typename T> MathMatrix<T>::MathMatrix(std::vector<std::vector<T>>& _raw_values)
			{
				this->invertor = new MatrixAlgorithm::MatrixInvertor::QRInvertor<T>();

				this->row_size = _raw_values.size();

				if (this->row_size != 0)
				{
					this->col_size = _raw_values.begin()->size();

					for (size_t index = 0; index < this->row_size; ++index)
					{
						this->values.push_back(MathVector<T>(_raw_values.at(index)));
					}

				}
				else
				{
					this->col_size = 0;
				}
			}

			template<typename T> MathMatrix<T>::MathMatrix(std::vector<MathVector<T>>& _values)
			{
				this->row_size = _values.size();
				this->invertor = new MatrixAlgorithm::MatrixInvertor::QRInvertor<T>();

				if (this->row_size != 0)
				{
					this->col_size = _values.begin()->getSize();

					this->values = _values;

				}
				else
				{
					this->col_size = 0;
				}
			}

			template<typename T> void MathMatrix<T>::setValues(std::vector<MathVector<T>>& _values)
			{
				this->row_size = _values.size();
				this->invertor = new MatrixAlgorithm::MatrixInvertor::QRInvertor<T>();

				this->values.clear();

				if (this->row_size != 0)
				{
					this->col_size = _values.begin()->getSize();

					this->values = _values;

				}
				else
				{
					this->col_size = 0;
					this->values = _values;
				}
			}
			//----------------------------------------------------------------------

			//--------------------------usual-operators------------------------------
			template<typename T>  size_t MathMatrix<T>::rows_size() const
			{
				return this->row_size;
			}

			template<typename T>  size_t MathMatrix<T>::cols_size() const
			{
				return this->col_size;
			}


			template<typename T>  MathMatrix<T>& MathMatrix<T>::clear()
			{
				this->row_size = 0;
				this->col_size = 0;

				this->values.clear();
			}

			template<typename T>  void MathMatrix<T>::push_back(MathVector<T> row)
			{
				if (row.getSize() != col_size)
				{
					throw std::out_of_range("");
				}
				else
				{
					this->row_size++;

					this->values.push_back(row);
				}
			}

			template<typename T> void MathMatrix<T>::pop_back()
			{
				if (this->row_size != 0)
				{
					this->row_size--;
					this->values.pop_back();
				}

				return;

			}

			template<typename T> void MathMatrix<T>::pop_at(size_t position)
			{
				if (this->row_size != 0)
				{
					if (position >= this->row_size || position < 0)
					{
						throw std::out_of_range("");
					}
					else
					{
						this->values.erase(this->values.begin() + position);
						this->col_size--;
					}

				}

				return;
			}

			template<typename T>  void MathMatrix<T>::set_invertor(MatrixAlgorithm::MatrixInvertor::MathMatrixInvertor<T> invertor)
			{
				this->invetor = &invertor;
			}

			template<typename T> void MathMatrix<T>::insert_element(T element, size_t rowIndex, size_t colIndex)
			{
				this->values.at(rowIndex).insert(element, colIndex);
			}

            template<typename T> MathVector<T> MathMatrix<T>::row(size_t rowIndex)
            {
                return this->values.at(rowIndex);
            }


			template<typename T> T MathMatrix<T>::at(size_t rowIndex, size_t colIndex)
			{
				return this->values.at(rowIndex).getElement(colIndex);
			}
			//----------------------------------------------------------------------

			//-------------------------operators------------------------------------
			template<typename T> MathMatrix<T>& MathMatrix<T>::operator=(const MathMatrix<T> & _MathMatrix)
			{
				this->col_size = _MathMatrix.col_size;
				this->row_size = _MathMatrix.row_size;

				this->values = _MathMatrix.values;

				this->invertor = _MathMatrix.invertor;

				return *this;
			}


			template<typename T> MathMatrix<T>& MathMatrix<T>::operator+(MathMatrix<T>& _other)
			{
				MathMatrix<T> *result = new MathMatrix<T>(*this);

				if (this->row_size != _other.row_size || this->col_size != _other.col_size)
				{
					throw std::logic_error("summarizing dimensions mismatch");
				}
				else
				{
					for (size_t index = 0; index < result->row_size(); ++index)
					{
						result->values.at(index) += _other.values.at(index);
					}
				}

				return *result;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator-(MathMatrix<T>& _other)
			{
				MathMatrix<T>* result = new MathMatrix<T>(*this);

				if (this->row_size != _other.row_size || this->col_size != _other.col_size)
				{
					throw std::logic_error("summarizing dimensions mismatch");
				}
				else
				{
					for (size_t index = 0; index < result->rows_size(); ++index)
					{
						result->values.at(index) -= _other.values.at(index);
					}
				}

				return *result;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::raw_multiply(MathMatrix<T>& _other)
			{
				if (this->col_size != _other.col_size)
				{
					throw std::exception();
				}
				else
				{
					size_t new_rows = this->row_size;
					size_t new_cols = _other.row_size;

					std::vector<std::vector<T>> raw_values(new_rows, std::vector<T>(col_size, 0));

					for (size_t rowindex = 0; rowindex < new_rows; ++rowindex)
					{
#pragma omp parallel for
						for (int colindex = 0; colindex < new_cols; ++colindex)
						{
							T value = this->values.at(rowindex) * _other.values.at(colindex);

							raw_values.at(rowindex).at(colindex) = value;
						}
					}

					return *(new MathMatrix<T>(raw_values));

				}
			}


			template<typename T> MathMatrix<T>& MathMatrix<T>::operator*(MathMatrix<T>& _other)
			{
				if (this->col_size != _other.row_size)
				{
					throw std::logic_error("summarizing dimensions mismatch");
				}
				else
				{
					size_t new_rows = this->row_size;
					size_t new_cols = _other.col_size;

					std::vector<std::vector<T>> raw_values(new_rows, std::vector<T>(col_size, 0));

					MathMatrix<T> _other_transponate = ~_other;

					for (size_t rowindex = 0; rowindex < new_rows; ++rowindex)
					{
#pragma omp parallel for
						for (int colindex = 0; colindex < new_cols; ++colindex)
						{
							T value = this->values.at(rowindex) * _other_transponate.values.at(colindex);

							raw_values.at(rowindex).at(colindex) = value;
						}
					}

					return  *(new MathMatrix<T>(raw_values));
				}
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator+=(MathMatrix<T>& _other)
			{

				if (this->row_size != _other.row_size || this->col_size != _other.col_size)
				{
					throw std::logic_error("summarizing dimensions mismatch");
				}
				else
				{
					for (size_t index = 0; index < this->row_size; ++index)
					{
						this->values.at(index) += _other.values.at(index);
					}
				}

				return *this;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator-=(MathMatrix<T>& _other)
			{
				if (this->row_size != _other.row_size || this->col_size != _other.col_size)
				{
					throw std::exception();
				}
				else
				{
					for (size_t index = 0; index < this->row_size(); ++index)
					{
						this->values.at(index) -= _other.values.at(index);
					}
				}

				return *this;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator*=(MathMatrix<T>& _other)
			{
				if (this->col_size != _other.row_size)
				{
					throw std::logic_error("summarizing dimensions mismatch");
				}
				else
				{
					size_t new_rows = this->row_size;
					size_t new_cols = _other.col_size;

					std::vector<std::vector<T>> raw_values(new_rows, std::vector<T>(col_size, 0));

					MathMatrix<T> _other_transponate = ~(*this);

					for (size_t rowindex = 0; rowindex < new_rows; ++rowindex)
					{
						for (size_t colindex = 0; colindex < new_cols; ++colindex)
						{
							T value = this->values.at(rowindex) * _other_transponate.values.at(colindex);

							raw_values.at(rowindex).at(colindex) = value;
						}
					}

					this->row_size = new_rows;
					this->col_size = new_cols;

					this->values = (MathMatrix<T>(raw_values)).values;

					return  *this;

				}
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator+(const T& _value)
			{

				MathMatrix<T>* result = new MathMatrix<T>(*this);

				for (size_t index = 0; index < result->row_size; ++index)
				{
					result->values.at(index) += _value;
				}

				return *result;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator-(const T& _value)
			{
				MathMatrix<T>* result = new MathMatrix<T>(*this);

				for (size_t index = 0; index < result->row_size; ++index)
				{
					result->values.at(index) -= _value;
				}

				return *result;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator*(const T& _value)
			{
				MathMatrix<T>* result = new MathMatrix<T>(*this);

				for (size_t index = 0; index < result->row_size; ++index)
				{
					result->values.at(index) *= _value;
				}

				return *result;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator/(const T& _value)
			{
				MathMatrix result(*this);

				for (size_t index = 0; index < result.row_size; ++index)
				{
					result.values.at(index) /= _value;
				}
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator+=(const T& _value)
			{
				for (size_t index = 0; index < this->row_size; ++index)
				{
					this->values.at(index) += _value;
				}

				return *this;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator-=(const T& _value)
			{
				for (size_t index = 0; index < this->row_size; ++index)
				{
					this->values.at(index) -= _value;
				}

				return *this;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator*=(const T& _value)
			{
				for (size_t index = 0; index < this->row_size; ++index)
				{
					this->values.at(index) *= _value;
				}

				return *this;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator/=(const T& _value)
			{
				for (size_t index = 0; index < this->row_size; ++index)
				{
					this->values.at(index) /= _value;
				}

				return *this;
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator!()
			{
				return this->invertor->invert(*this);
			}

			template<typename T> MathMatrix<T>& MathMatrix<T>::operator~()
			{
				size_t row_size_t = this->col_size;
				size_t col_size_t = this->row_size;

				std::vector<MathVector<T>> values_t(this->col_size, MathVector<T>(this->row_size, 0));

				for (size_t rowindex = 0; rowindex < this->row_size; ++rowindex)
				{
					typename MathVector<T>::fast_iterator  it = this->values.at(rowindex).fast_begin();
					typename MathVector<T>::fast_iterator  end = this->values.at(rowindex).fast_end();
					for (; it != end; ++it)
					{
						values_t.at(it.index()).insert(it.getElem(), rowindex);
					}
				}

				MathMatrix<T>* result = new MathMatrix<T>(values_t);

				return *result;
			}

			template<typename T> bool MathMatrix<T>::operator==(MathMatrix &_other) const
			{
				if (this->row_size != _other.row_size || this->col_size != _other.col_size)
				{
					return false;
				}
				else
				{

					for (size_t index = 0; index < this->row_size; ++index)
					{

						if (this->values.at(index) != _other.values.at(index))
						{
							return false;
						}

					}

					return true;
				}

				return false;
			}

			template<typename T> bool MathMatrix<T>::operator!=(MathMatrix &_other) const
			{
				if (this->row_size == _other.row_size && this->col_size == _other.col_size)
				{
					return false;
				}
				else
				{

					for (size_t index = 0; index < this->row_size; ++index)
					{

						if (this->values.at(index) == _other.values.at(index))
						{
							return false;
						}

					}

					return true;
				}

				return false;
			}

			template<typename T> MathVector<T>& MathMatrix<T>::operator[](size_t _row_index)
			{
				if (_row_index >= this->row_size || _row_index < 0)
				{
					throw std::out_of_range("");
				}
				else
				{
					return this->values.at(_row_index);
				}
			}
			//----------------------------------------------------------------------
		}
	}
}

#endif //MATHMATRIX_H


