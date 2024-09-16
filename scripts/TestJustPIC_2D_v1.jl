using CairoMakie, MAT, Printf
using Base.Threads
using JustPIC, JustPIC._2D
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

const backend = JustPIC.CPUBackend 
USE_GPU  = false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@parallel_indices (I...) function InitialFieldsParticles!( phases, px, py, index)
    @inbounds for ip in cellaxes(phases)
        # quick escape
        @cell(index[ip, I...]) == 0 && continue
        x = @cell px[ip, I...]
        y = @cell py[ip, I...]
        if x<y
            @cell phases[ip, I...] = 1.0
        else
            @cell phases[ip, I...] = 2.0
        end
    end
    return nothing
end

function main()

    @printf("Running on %d thread(s)\n", nthreads())
    L  = (x=1., y=1.)
    Nc = (x=40, y=40  )
    Nv = (x=Nc.x+1,   y=Nc.x+1  )
    Δ  = (x=L.x/Nc.x, y=L.y/Nc.y  )
    Nt   = 40000
    Nout = 5000
    C    = 0.25

    verts     = (x=LinRange(0, L.x, Nv.x), y=LinRange(0, L.y, Nv.y))
    cents_ext = (x=LinRange(-Δ.x/2, L.x+Δ.x/2, Nc.x+2), y=LinRange(-Δ.y/2, L.y+Δ.y/2, Nc.y+2))
    cents     = (x=LinRange(+Δ.x/2, L.x-Δ.x/2, Nc.x  ), y=LinRange(+Δ.y/2, L.y-Δ.y/2, Nc.y  ))

    # Import data
    file = matopen("data/CornerFlow2D.mat")
    Vx   = read(file, "Vx") # ACHTUNG THIS CONTAINS GHOST NODES AT NORTH/SOUTH
    Vy   = read(file, "Vy") # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
    Pt   = read(file, "Pt")
    close(file)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 50, 5
    particles = init_particles(
        backend, 
        nxcell, 
        max_xcell,
        min_xcell, 
        values(verts),
        values(Δ),
        values(Nc)
    ) # random position by default

    # Initialise phase field
    particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

    @parallel InitialFieldsParticles!(phases, particles.coords..., particles.index)

    # Time step
    Δt = C * min(Δ...) / max(maximum(Vx), maximum(Vy))
    @show Δt

    # Create necessary tuples
    V       = (Vx, Vy)
    grid_vx = (verts.x, cents_ext.y)
    grid_vy = (cents_ext.x, verts.y)

    for it=1:Nt

        advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy), Δt)
        # advection_LinP!(particles, RungeKutta2(), V, (grid_vx, grid_vy), Δt)
        # advection_MQS!(particles, RungeKutta2(), V, (grid_vx, grid_vy), Δt)
        move_particles!(particles, values(verts), particle_args)        
        # inject_particles!(particles, particle_args, values(verts)) 

        if mod(it,Nout) == 0 || it==1

            @show Npart = sum(particles.index.data)
            particle_density = [sum(p) for p in particles.index]

            # Plots
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = phases.data[:]
            idxv = particles.index.data[:]
            
            f = Figure()
            ax1 = Axis(f[1, 1], title="Particles", aspect=1.0)
            ax2 = Axis(f[1, 2], title="Density", aspect=1.0)

            scatter!(ax1, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)

            # f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=2)
            # f,ax,h=heatmap(cents.x, cents.y, Vmag[:,:,1])
            # f,ax,h=arrows(cents.x, cents.y, Vxc[:,:,40]./Vmag[:,:,40], Vyc[:,:,40]./Vmag[:,:,20], arrowsize = 5, lengthscale = 1e-2)
            # f,ax,h=arrows(cents.x, cents.z, Vxc[:,:,20]./Vmag[:,:,2], Vzc[:,:,2]./Vmag[:,:,2], arrowsize = 5, lengthscale = 1e-2)
            hm = heatmap!(ax2, cents.x, cents.y, particle_density[:,:], colormap=:inferno, colorrange=(0., 35.))
            Colorbar(f[1, 3], hm)
            display(f)

        end
    end
end 

main()
