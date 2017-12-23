#pragma once
#include "stdafx.h"
#include "nnet_loss.h"
#include "nnet_layer.h"


namespace nnet
{
  class bpnn
  {
  public:
    field<layer> layers;
    field<mat> ios;
    field<mat> weights;
    field<mat> dlosses;
    field<mat> deltas;
    field<mat> biases;
    field<mat> alphas;
    field<mat> lambdas;
    field<loss*> loss_itfs;
    int n_layers = 0;


    const mat &output() { return ios(n_layers); }


    // ios: layer-io of nnet, if n_layers = N, then n_ios = N + 1
    // layers: not include input-layer
    void init_malloc(const uvec &ios_dim)
    {
      n_layers = ios_dim.n_elem - 1;
      layers.set_size(n_layers);
      ios.set_size(ios_dim.n_elem);
      weights.set_size(n_layers);
      dlosses.set_size(ios_dim.n_elem);
      deltas.set_size(n_layers);
      alphas.set_size(n_layers);
      lambdas.set_size(n_layers);
      biases.set_size(n_layers);
      loss_itfs.set_size(ios_dim(ios_dim.n_elem - 1));
      weight_gradients.set_size(n_layers);
      bias_gradients.set_size(n_layers);

      loss_itfs.fill(NULL);
      int i = 0;
      weights.for_each(
        [&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), ios_dim(i)); ++i; }
      );
      auto mat_malloc = [&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), 1); ++i; };
      i = 0;
      biases.for_each(mat_malloc);
      i = 0;
      alphas.for_each(mat_malloc);
      i = 0;
      lambdas.for_each(mat_malloc);
    }


    void train(
      const mat &tx,
      const mat &ty,
      const int &n_iterations,
      mat *tlosses = NULL,
      const mat *vx = NULL,
      const mat *vy = NULL,
      mat *vlosses = NULL)
    {
      loss_itfs.for_each(
        [](loss* &f) { assert(f != NULL); }
      );

      int len = ty.n_cols, out_dim = ty.n_rows;

      if (tlosses != NULL)
        tlosses->set_size(n_iterations, out_dim);
      if (vx != NULL && vy != NULL && vlosses != NULL)
        vlosses->set_size(n_iterations, out_dim);

      system("cls");
      std::cout << "0 %" << std::endl;
      std::cout << "0 / " << n_iterations << std::endl;

      for (int k = 1; k <= n_iterations; ++k)
      {
        if (!(k % 10))
        {
          system("cls");
          std::cout << k * 100 / n_iterations << " %" << std::endl;
          std::cout << k << " / " << n_iterations << std::endl;
        }

        // propagate
        propagate(tx);

        // dloss(l + 1)
        int i = 0;
        if (dlosses(n_layers).n_rows != ty.n_rows || dlosses(n_layers).n_cols != ty.n_cols)
          dlosses(n_layers).set_size(ty.n_rows, ty.n_cols);
        dlosses(n_layers).each_row(
          [&i, this, &ty](mat &r) { this->loss_itfs(i)->diff(this->ios(this->n_layers).row(i), ty.row(i), &r); ++i; }
        );

        // back propagate
        back_propagate(k);

        // k-th iteration's train avg-loss
        i = 0;
        if (tlosses != NULL)
        {
          propagate(tx);
          tlosses->row(k - 1).each_col(
            [&i, this, &ty](mat &r) { this->loss_itfs(i)->avg_eval(this->ios(this->n_layers).row(i), ty.row(i), &r); ++i; }
          );
        }

        // k-th iteration's validation avg-loss
        i = 0;
        if (vx != NULL && vy != NULL && vlosses != NULL)
        {
          propagate(*vx);
          vlosses->row(k - 1).each_col(
            [&i, this, vy](mat &r) { this->loss_itfs(i)->avg_eval(this->ios(this->n_layers).row(i), vy->row(i), &r); ++i; }
          );
        }
      }

      system("cls");
      std::cout << "100 %" << std::endl;
      std::cout << n_iterations << " / " << n_iterations << std::endl;
    }


    void propagate(const mat &x)
    {
      // init ios(0), first input
      ios(0) = x;

      // propagate
      for (int l = 0; l < n_layers; ++l)
        layers(l).propagate(
          ios(l),
          weights(l),
          biases(l),
          &ios(l + 1)
        );
    }


  private:
    field<mat> weight_gradients;
    field<mat> bias_gradients;

    void back_propagate(const int &k)
    {
      for (int l = n_layers - 1; l >= 0; --l)
        layers(l).back_propagate(
          ios(l),
          ios(l + 1),
          dlosses(l + 1),
          alphas(l),
          lambdas(l),
          k,
          &dlosses(l),
          &deltas(l),
          &weight_gradients(l),
          &bias_gradients(l),
          &weights(l),
          &biases(l)
        );
    }
  };
}
